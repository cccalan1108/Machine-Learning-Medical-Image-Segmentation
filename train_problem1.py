import torch
import torch.optim as optim
import wandb
import os
import argparse
import numpy as np 
from torch.utils.data import DataLoader, Subset

from src.helpers import enforce_reproducibility
from src.data_loader import SegmentationData, ImageProcessor
from src.networks import MedicalResUNet
from src.objectives import SegmentationCompoundLoss 
from src.engine import ExecutionManager
from src.helpers import ExperimentConfig

CFG = ExperimentConfig(
    project_name="Medical_Seg_Ensemble",
    seed=42,
    img_size=(256, 256), 
    batch_size=16,       
    epochs=150,          
    lr=3e-4,             
    weight_decay=1e-4,
    model_name="ResUNet_Fold"
)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0-4)")
    return parser.parse_args()

def main():
    args = get_args()
    fold_idx = args.fold
    enforce_reproducibility(CFG.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{CFG.model_name}_{fold_idx}"
    wandb.init(project=CFG.project_name, config=CFG.get_dict(), name=run_name, reinit=True)
    train_proc = ImageProcessor(target_size=CFG.img_size, augment=True)
    val_proc = ImageProcessor(target_size=CFG.img_size, augment=False)
    full_dataset = SegmentationData(root_dir="./dataset", mode="train", transform_processor=None) 

    print(f"Generating splits for Fold {fold_idx} ")
    all_indices = np.arange(len(full_dataset))
    
    rng = np.random.default_rng(CFG.seed)
    rng.shuffle(all_indices)
    
    folds = np.array_split(all_indices, 5)
    
    val_idx = folds[fold_idx]
    train_folds = [folds[i] for i in range(5) if i != fold_idx]
    train_idx = np.concatenate(train_folds)
    
    train_idx = train_idx.tolist()
    val_idx = val_idx.tolist()
    
    print(f"Starting Fold {fold_idx}: Train={len(train_idx)}, Val={len(val_idx)}")
    
    train_set = Subset(SegmentationData(root_dir="./dataset", mode="train", transform_processor=train_proc), train_idx)
    val_set = Subset(SegmentationData(root_dir="./dataset", mode="train", transform_processor=val_proc), val_idx)
    
    train_loader = DataLoader(train_set, batch_size=CFG.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=CFG.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = MedicalResUNet(in_channels=3, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=1e-6)
    
    criterion = SegmentationCompoundLoss() 
    
    manager = ExecutionManager(model, optimizer, criterion, scheduler, device, CFG)
    
    manager.execute(train_loader, val_loader, CFG.epochs)
    
    wandb.finish()
    
    if os.path.exists("checkpoints/best_model.pth"):
        target_name = f"checkpoints/best_model_fold_{fold_idx}.pth"
        if os.path.exists(target_name):
            os.remove(target_name)
        os.rename("checkpoints/best_model.pth", target_name)
        print(f"Saved: {target_name}")

if __name__ == "__main__":
    main()