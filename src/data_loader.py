import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms
from typing import List, Tuple, Optional
class SelfSupervisedInpaintingDataset(Dataset):
    def __init__(self, root_dirs: List[str], img_size=(256, 256), mask_ratio=0.75, patch_size=16):
        self.img_paths = []
        for d in root_dirs:
            self.img_paths.extend(glob.glob(os.path.join(d, "*.tif"))) 
        
        self.resize = transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size

    def __len__(self):
        return len(self.img_paths)

    def _apply_patch_masking(self, img: torch.Tensor):
        _, h, w = img.shape
        num_patches_h = h // self.patch_size
        num_patches_w = w // self.patch_size
        num_patches = num_patches_h * num_patches_w
        
        num_masked = int(num_patches * self.mask_ratio)
        mask_indices = np.random.choice(num_patches, num_masked, replace=False)
        
        mask = torch.ones((h, w), device=img.device)
        
        for idx in mask_indices:
            ph = idx // num_patches_w
            pw = idx % num_patches_w
            top = ph * self.patch_size
            left = pw * self.patch_size
            mask[top:top+self.patch_size, left:left+self.patch_size] = 0.0
            
        masked_img = img * mask
        return masked_img, mask

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")
        img = TF.to_tensor(img)
        img = self.resize(img)
        
        masked_input, mask_map = self._apply_patch_masking(img.clone())
        
        return {
            "image": masked_input, 
            "target": img,   
            "mask": mask_map.unsqueeze(0) 
        }

class ImageProcessor:
    def __init__(self, target_size: Tuple[int, int] = (256, 256), augment: bool = False):
        self.target_size = target_size
        self.augment = augment

    def _strong_augmentation(self, image: torch.Tensor, label: torch.Tensor):
        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)
        if random.random() > 0.5:
            image = TF.vflip(image)
            label = TF.vflip(label)
            
        if random.random() > 0.3:
            angle = random.uniform(-10, 10)
            scale = random.uniform(0.85, 1.15)
            translate = (random.uniform(-0.1, 0.1) * image.shape[2], 
                         random.uniform(-0.1, 0.1) * image.shape[1])
            image = TF.affine(image, angle=angle, translate=translate, scale=scale, shear=5, interpolation=transforms.InterpolationMode.BILINEAR)
            label = TF.affine(label, angle=angle, translate=translate, scale=scale, shear=5, interpolation=transforms.InterpolationMode.NEAREST)

        if random.random() > 0.2:
            if random.random() > 0.5:
                blur = transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 1.0))
                image = blur(image)
            if random.random() > 0.5:
                noise = torch.randn_like(image) * 0.03
                image = image + noise
                image = torch.clamp(image, 0, 1)

        if random.random() > 0.2:
            image = TF.adjust_brightness(image, random.uniform(0.7, 1.3))
            image = TF.adjust_contrast(image, random.uniform(0.7, 1.3))
            image = TF.adjust_gamma(image, random.uniform(0.8, 1.2))

        return image, label

    def __call__(self, image_path: str, label_path: Optional[str] = None):
        img = Image.open(image_path).convert("RGB") 
        img = TF.to_tensor(img) 
        
        if label_path:
            lbl = Image.open(label_path).convert("L") 
            lbl = TF.to_tensor(lbl) 
        else:
            lbl = None

        img = TF.resize(img, self.target_size, interpolation=transforms.InterpolationMode.BILINEAR)
        if lbl is not None:
            lbl = TF.resize(lbl, self.target_size, interpolation=transforms.InterpolationMode.NEAREST)

        if self.augment and lbl is not None:
            img, lbl = self._strong_augmentation(img, lbl)

        return img, lbl

class SegmentationData(Dataset):
    def __init__(self, root_dir: str, mode: str = "train", transform_processor=None):
        self.mode = mode
        self.processor = transform_processor
        
        if mode == "train" or mode == "val":
            self.img_dir = os.path.join(root_dir, "train", "image")
            self.lbl_dir = os.path.join(root_dir, "train", "label")
            self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.tif"))) 
        else:
            self.img_paths = sorted(glob.glob(os.path.join(root_dir, mode, "image", "*.tif")))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        if self.mode in ["train", "val"]:
            fname = os.path.basename(img_path)
            lbl_path = os.path.join(self.lbl_dir, fname.replace("image", "label")) 
            img, lbl = self.processor(img_path, lbl_path)
            return {"image": img, "label": lbl, "filename": fname}
        else:
            img, _ = self.processor(img_path, None)
            return {"image": img, "filename": os.path.basename(img_path)}