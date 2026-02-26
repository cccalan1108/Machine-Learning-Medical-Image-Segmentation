import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.helpers import enforce_reproducibility, save_checkpoint
from src.data_loader import SegmentationData, SelfSupervisedInpaintingDataset, ImageProcessor
from src.networks import HybridViTSegmenter 
from src.objectives import SegmentationCompoundLoss
from src.engine import ExecutionManager
from src.helpers import ExperimentConfig

SSL_CFG = ExperimentConfig(
    project_name="Medical_Seg_Problem3_SSL_Ultimate",
    seed=42,
    img_size=(256, 256),
    batch_size=16,      
    accum_iter=4,        
    epochs=100,          
    lr=5e-4,             
    weight_decay=0.05,   
    model_name="MAE_Pretrain"
)

FT_CFG = ExperimentConfig(
    project_name="Medical_Seg_Problem3_Finetune_Ultimate",
    seed=42,
    img_size=(256, 256),
    batch_size=8,       
    accum_iter=4,       
    epochs=200,          
    lr=1e-4,             
    weight_decay=0.05,
    model_name="MAE_Finetune_Best"
)


class DeepSupervisionManager(ExecutionManager):
    def run_epoch(self, loader, is_train=True):
        self.model.train() if is_train else self.model.eval()
        total_loss = 0
        from tqdm import tqdm
        from src.helpers import move_batch_to_device
        
        pbar = tqdm(loader, desc="Training" if is_train else "Validating", leave=False)
        all_preds, all_labels = [], []
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            batch = move_batch_to_device(batch, self.device)
            if "target" in batch: inputs, labels = batch['image'], batch['target'] 
            else: inputs, labels = batch['image'], batch['label'] 

            with torch.set_grad_enabled(is_train):
                with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                    outputs = self.model(inputs)
                    if is_train:
                        if isinstance(outputs, list):
                            if "target" in batch: 
                                loss = self.criterion(outputs[0], labels)
                            else: # Seg
                                loss = 0.6 * self.criterion(outputs[0], labels) + \
                                       0.2 * self.criterion(outputs[1], labels) + \
                                       0.2 * self.criterion(outputs[2], labels)
                        else:
                            loss = self.criterion(outputs, labels)
                        loss = loss / self.config.accum_iter
                    else:
                        if isinstance(outputs, list): outputs = outputs[0]
                        loss = self.criterion(outputs, labels)

                if is_train:
                    self.scaler.scale(loss).backward()
                    if ((batch_idx + 1) % self.config.accum_iter == 0) or (batch_idx + 1 == len(loader)):
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                    total_loss += loss.item() * self.config.accum_iter
                else:
                    total_loss += loss.item()

            if not is_train and "label" in batch:
                if isinstance(outputs, list): outputs = outputs[0]
                all_preds.append(outputs.detach().cpu())
                all_labels.append(labels.detach().cpu())

        avg_loss = total_loss / len(loader) if is_train else 0
        metric_score = 0
        if not is_train and all_preds and "label" in batch:
            from src.evaluator import SorensenDiceMetric
            metric_score = SorensenDiceMetric()(torch.cat(all_preds), torch.cat(all_labels)).item()

        return avg_loss, metric_score

    def execute(self, train_loader, val_loader, epochs):
        print(f"Starting training for {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            train_loss, _ = self.run_epoch(train_loader, is_train=True)
            val_loss, val_dice = self.run_epoch(val_loader, is_train=False)
            

            if self.scheduler:
                self.scheduler.step()

            print(f"Epoch {epoch}/{epochs} | Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}")
            wandb.log({"train_loss": train_loss, "val_dice": val_dice, "epoch": epoch})


            if val_dice > self.best_score:
                self.best_score = val_dice
                save_checkpoint(self.model.state_dict(), "checkpoints", "best_model_p3.pth")
                print(f"New best score! Model saved to checkpoints/best_model_p3.pth")

def run_pretraining(device):
    print(f"\n[Phase 1] Starting MAE Pretraining...")
    wandb.init(project=SSL_CFG.project_name, config=SSL_CFG.get_dict(), name=SSL_CFG.model_name, reinit=True)
    
    data_dirs = ["./dataset/train/image", "./dataset/public/image"]
    ssl_dataset = SelfSupervisedInpaintingDataset(data_dirs, img_size=SSL_CFG.img_size, mask_ratio=0.75)
    ssl_loader = DataLoader(ssl_dataset, batch_size=SSL_CFG.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    model = HybridViTSegmenter(
        in_channels=3, num_classes=3, deep_supervision=True,
        img_size=256, embed_dim=512, depth=4, heads=8
    ).to(device)
    
    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=SSL_CFG.lr, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=SSL_CFG.epochs)
    
    manager = DeepSupervisionManager(model, optimizer, criterion, scheduler, device, SSL_CFG)
    
    model.train()
    for epoch in range(1, SSL_CFG.epochs + 1):
        avg_loss, _ = manager.run_epoch(ssl_loader, is_train=True)
        scheduler.step()
        print(f"[SSL] Epoch {epoch}/{SSL_CFG.epochs} | MAE Recon Loss: {avg_loss:.5f}")
        wandb.log({"ssl_loss": avg_loss})
    
    save_path = "checkpoints/mae_pretrained.pth"
    save_checkpoint(model.state_dict(), "checkpoints", "mae_pretrained.pth")
    wandb.finish()
    return save_path

def run_finetuning(device, pretrained_path):
    print(f"\n[Phase 2] Starting MAE Fine-tuning...")
    wandb.init(project=FT_CFG.project_name, config=FT_CFG.get_dict(), name=FT_CFG.model_name, reinit=True)
    
    train_proc = ImageProcessor(target_size=FT_CFG.img_size, augment=True)
    val_proc = ImageProcessor(target_size=FT_CFG.img_size, augment=False)
    train_loader = DataLoader(SegmentationData("./dataset", "train", train_proc), batch_size=FT_CFG.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(SegmentationData("./dataset", "val", val_proc), batch_size=FT_CFG.batch_size, shuffle=False, num_workers=4)
    
    model = HybridViTSegmenter(
        in_channels=3, num_classes=1, deep_supervision=True,
        img_size=256, embed_dim=512, depth=4, heads=8
    ).to(device)
    
    print(f"Loading MAE weights...")
    pretrained_dict = torch.load(pretrained_path, map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                       if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print(f"Weights loaded. Starting Fine-tuning.")
    
    optimizer = optim.AdamW(model.parameters(), lr=FT_CFG.lr, weight_decay=FT_CFG.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=FT_CFG.epochs, eta_min=1e-6)
    criterion = SegmentationCompoundLoss() 
    
    manager = DeepSupervisionManager(model, optimizer, criterion, scheduler, device, FT_CFG)
    manager.execute(train_loader, val_loader, FT_CFG.epochs) 
    
    wandb.finish()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists("checkpoints"): os.makedirs("checkpoints")
    

    #pretrained_path = run_pretraining(device) 
    

    pretrained_path = "checkpoints/mae_pretrained.pth"

    run_finetuning(device, pretrained_path)

if __name__ == "__main__":
    main()