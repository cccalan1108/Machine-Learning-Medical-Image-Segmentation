import torch
import torch.nn as nn
from tqdm import tqdm
from src.helpers import move_batch_to_device

class ExecutionManager:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        config
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.best_score = 0.0
        self.scaler = torch.cuda.amp.GradScaler()

    def run_epoch(self, loader, is_train=True):
        self.model.train() if is_train else self.model.eval()
        total_loss = 0
        
        pbar = tqdm(loader, desc="Training" if is_train else "Validating", leave=False)
        all_preds = []
        all_labels = []

        self.optimizer.zero_grad()

        for batch in pbar:
            batch = move_batch_to_device(batch, self.device)
            images = batch['image']
            labels = batch.get('label') 

            with torch.set_grad_enabled(is_train):
                with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda')):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels) if labels is not None else 0

                if is_train:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    total_loss += loss.item()
                else:
                    total_loss += loss.item()

            if not is_train and labels is not None:
                all_preds.append(outputs.detach().cpu())
                all_labels.append(labels.detach().cpu())

        avg_loss = total_loss / len(loader) if is_train else 0
        
        metric_score = 0
        if not is_train and all_preds:
            from src.evaluator import SorensenDiceMetric
            metric_score = SorensenDiceMetric()(torch.cat(all_preds), torch.cat(all_labels)).item()

        return avg_loss, metric_score

    def execute(self, train_loader, val_loader, epochs):
        pass