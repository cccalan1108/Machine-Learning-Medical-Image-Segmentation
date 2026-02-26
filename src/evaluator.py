import torch

class SorensenDiceMetric:
    def __init__(self, epsilon: float = 1e-6, activation_threshold: float = 0.5):
        self.epsilon = epsilon
        self.threshold = activation_threshold

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > self.threshold).float()

            preds_flat = preds.view(preds.size(0), -1)
            targets_flat = targets.view(targets.size(0), -1)

            intersection = (preds_flat * targets_flat).sum(dim=1)
            cardinality = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)
            dice_score = (2.0 * intersection + self.epsilon) / (cardinality + self.epsilon)
            
            return dice_score.mean()