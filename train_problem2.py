import torch
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.helpers import enforce_reproducibility
from src.data_loader import SegmentationData, ImageProcessor
from src.networks import HybridViTSegmenter 
from src.objectives import SegmentationCompoundLoss
from src.engine import ExecutionManager
from src.helpers import ExperimentConfig

CFG = ExperimentConfig(
    project_name="Medical_Seg_Problem2_Stable",
    seed=42,
    img_size=(256, 256),
    batch_size=8,
    accum_iter=4,
    epochs=150,
    lr=1e-4,
    weight_decay=0.05,
    model_name="HybridViT_Stable"
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
            images, labels = batch['image'], batch['label']

            with torch.set_grad_enabled(is_train):
                with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                    outputs = self.model(images)
                    
                    if is_train:
                        if isinstance(outputs, list):
                            loss_final = self.criterion(outputs[0], labels)
                            loss_ds1 = self.criterion(outputs[1], labels)
                            loss_ds2 = self.criterion(outputs[2], labels)
                            loss = 0.6 * loss_final + 0.2 * loss_ds1 + 0.2 * loss_ds2
                        else:
                            loss = self.criterion(outputs, labels)
                            
                        loss = loss / self.config.accum_iter
                    else:
                        if isinstance(outputs, list):
                            outputs = outputs[0]
                        
                        loss = self.criterion(outputs, labels)

                if is_train:
                    self.scaler.scale(loss).backward()
                    if ((batch_idx + 1) % self.config.accum_iter == 0) or (batch_idx + 1 == len(loader)):
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad()
                    total_loss += loss.item() * self.config.accum_iter
                else:
                    total_loss += loss.item()

            if not is_train and labels is not None:
                if isinstance(outputs, list):
                    outputs = outputs[0]
                all_preds.append(outputs.detach().cpu())
                all_labels.append(labels.detach().cpu())

        avg_loss = total_loss / len(loader) if is_train else 0
        metric_score = 0
        if not is_train and all_preds:
            from src.evaluator import SorensenDiceMetric
            metric_score = SorensenDiceMetric()(torch.cat(all_preds), torch.cat(all_labels)).item()

        return avg_loss, metric_score

def main():
    enforce_reproducibility(CFG.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project=CFG.project_name, config=CFG.get_dict(), name=CFG.model_name)
    
    train_proc = ImageProcessor(target_size=CFG.img_size, augment=True)
    val_proc = ImageProcessor(target_size=CFG.img_size, augment=False)
    
    train_set = SegmentationData(root_dir="./dataset", mode="train", transform_processor=train_proc)
    val_set = SegmentationData(root_dir="./dataset", mode="val", transform_processor=val_proc)
    
    train_loader = DataLoader(train_set, batch_size=CFG.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=CFG.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    model = HybridViTSegmenter(
        img_size=CFG.img_size[0],
        patch_size=16,
        embed_dim=512, 
        depth=4,
        heads=8,
        num_classes=1,
        deep_supervision=True
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=1e-6)
    criterion = SegmentationCompoundLoss() 
    
    manager = DeepSupervisionManager(model, optimizer, criterion, scheduler, device, CFG)
    manager.execute(train_loader, val_loader, CFG.epochs)
    
    wandb.finish()

if __name__ == "__main__":
    main()