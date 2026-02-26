import os
import glob
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import functional as TF
from src.networks import HybridViTSegmenter

CHECKPOINT_PATH = "checkpoints/p3/best_model_p3.pth" 
PUBLIC_DIR = "dataset/public/image"
PRIVATE_DIR = "dataset/private/image"
OUTPUT_CSV = "submission.csv"
IMG_SIZE = (256, 256)

THRESHOLD = 0.5        
MIN_PIXEL_SIZE = 50    

def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def predict_with_tta(model, img_tensor):
    pred_1 = torch.sigmoid(model(img_tensor))
    
    pred_2 = torch.sigmoid(model(TF.hflip(img_tensor)))
    pred_2 = TF.hflip(pred_2)
    
    pred_3 = torch.sigmoid(model(TF.vflip(img_tensor)))
    pred_3 = TF.vflip(pred_3)
    
    return (pred_1 + pred_2 + pred_3) / 3.0

def predict_and_process(model, image_paths, device):
    results = []
    model.eval()
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Predicting P3 (TTA+Filter)"):
            filename = os.path.basename(img_path)
            
            original_img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = original_img.size
            
            img_tensor = TF.to_tensor(TF.resize(original_img, IMG_SIZE, interpolation=TF.InterpolationMode.BILINEAR))
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            probs = predict_with_tta(model, img_tensor)
            
            probs_resized = F.interpolate(probs, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            
            pred_mask = (probs_resized > THRESHOLD).squeeze().cpu().numpy().astype(np.uint8)
            
            if pred_mask.sum() < MIN_PIXEL_SIZE:
                pred_mask[:] = 0
            
            if pred_mask.sum() == 0:
                rle_str = "1 0"
            else:
                rle_str = rle_encode(pred_mask)
            
            results.append({
                "row ID": filename,
                "label": rle_str
            })
            
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = HybridViTSegmenter(
        img_size=256,
        patch_size=16,
        embed_dim=512,
        depth=4,
        heads=8,
        num_classes=1,
        deep_supervision=False
    ).to(device)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"找不到權重: {CHECKPOINT_PATH}")
    
    print(f"Loading weights from {CHECKPOINT_PATH}...")
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)

    public_files = sorted(glob.glob(os.path.join(PUBLIC_DIR, "*.tif")))
    private_files = sorted(glob.glob(os.path.join(PRIVATE_DIR, "*.tif")))
    all_files = public_files + private_files

    predictions = predict_and_process(model, all_files, device)
    
    df = pd.DataFrame(predictions)

    df['label'] = df['label'].fillna("1 0")
    df.loc[df['label'] == '', 'label'] = "1 0"
    
    df.to_csv(OUTPUT_CSV, index=False) 
    print(f"Submission saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()