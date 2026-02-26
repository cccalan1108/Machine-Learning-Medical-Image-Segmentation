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

from src.networks import MedicalResUNet

MODEL_PATHS = [
    "checkpoints/p1/best_model_fold_0.pth",
    "checkpoints/p1/best_model_fold_1.pth",
    "checkpoints/p1/best_model_fold_2.pth",
    "checkpoints/p1/best_model_fold_3.pth",
    "checkpoints/p1/best_model_fold_4.pth"
]

IMG_SIZE = (256, 256)
THRESHOLD = 0.35  
OUTPUT_CSV = "submission.csv"
def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Problem 1 (5-Fold Ensemble + TTA)")
    parser.add_argument("public_dir", type=str, help="Path to public dataset directory")
    parser.add_argument("private_dir", type=str, help="Path to private dataset directory")
    return parser.parse_args()

def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def load_ensemble_models(device):
    models = []
    print(f"Loading {len(MODEL_PATHS)} models for ensemble...")
    
    for i, path in enumerate(MODEL_PATHS):
        if not os.path.exists(path):
            print(f"Warning: Model path not found: {path}. Skipping this fold.")
            continue
            
        model = MedicalResUNet(in_channels=3, num_classes=1).to(device)
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
        print(f"Loaded Fold {i}: {path}")
        
    if len(models) == 0:
        raise RuntimeError("No models loaded! Check your checkpoint paths.")
        
    return models

def predict_with_tta(model, img_tensor):
    pred_orig = torch.sigmoid(model(img_tensor))
    
    img_h = TF.hflip(img_tensor)
    pred_h = torch.sigmoid(model(img_h))
    pred_h = TF.hflip(pred_h) 
    
    img_v = TF.vflip(img_tensor)
    pred_v = torch.sigmoid(model(img_v))
    pred_v = TF.vflip(pred_v) 
    
    return (pred_orig + pred_h + pred_v) / 3.0

def predict_ensemble(models, image_paths, device):
    results = []
    
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Ensemble + TTA Predicting"):
            filename = os.path.basename(img_path)
            
            original_img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = original_img.size
            
            img_tensor = TF.to_tensor(TF.resize(original_img, IMG_SIZE, interpolation=TF.InterpolationMode.BILINEAR))
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            avg_probs = torch.zeros((1, 1, 256, 256)).to(device)
            
            for model in models:
                probs = predict_with_tta(model, img_tensor)
                avg_probs += probs
            
            avg_probs /= len(models)
            
            probs_resized = F.interpolate(avg_probs, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            
            pred_mask = (probs_resized > THRESHOLD).squeeze().cpu().numpy().astype(np.uint8)
            
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
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models = load_ensemble_models(device)

    public_files = sorted(glob.glob(os.path.join(args.public_dir, "*.tif")))
    private_files = sorted(glob.glob(os.path.join(args.private_dir, "*.tif")))
    all_files = public_files + private_files
    
    print(f"Found {len(public_files)} public + {len(private_files)} private images. Total: {len(all_files)}")

    predictions = predict_ensemble(models, all_files, device)

    df = pd.DataFrame(predictions)
    
    df['label'] = df['label'].fillna("1 0")
    df.loc[df['label'] == '', 'label'] = "1 0"
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Submission saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()