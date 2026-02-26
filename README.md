[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/uOK6onxy)
# ML Assignment 3: Medical Image Segmentation

This repository contains the implementation for Assignment 3 of the Machine Learning (CSIE5043) course, Fall 2025, at National Taiwan University. The goal of this assignment is to develop and evaluate segmentation models for medical images using different neural network architectures and training strategies.

The dataset is organized into three folders: train, public, and private. The train folder contains two subfolders, image and label, while the public and private folders contain only images. Each image file name consists of a six-digit number, where the first three digits represent the patient ID and the last three digits represent the slice ID.

Submissions are evaluated using the mean Dice Similarity Coefficient (DSC). The Dice coefficient measures the pixel-wise overlap between the predicted segmentation and the ground truth, and is defined as:

$$
\text{Dice}(X,Y) = \frac{2|X \cap Y|}{|X| + |Y|}
$$

where X is the set of predicted pixels and Y is the set of ground-truth pixels. If both X and Y are empty, the Dice coefficient is defined as 1.


## 🧩 Setup
To install all required dependencies, simply run:
```
uv sync
```
> [!TIP]
> Ensure that uv is installed beforehand: `pip install uv`
If you wish to use any additional packages, please obtain approval from the TA beforehand.




## 🏋️ Training
To start training:
```
bash script/run_train.sh
```
(Example for Problem 1 with 5-Fold Cross-Validation):
Train 5 separate models for ensemble
python train_problem1.py --fold 0
python train_problem1.py --fold 1
python train_problem1.py --fold 2
python train_problem1.py --fold 3
python train_problem1.py --fold 4

Problem 2: Transformer + UNet (Nested-TransUNet)
Trains the model from scratch using Deep Supervision.
python train_problem2.py

Problem 3: Self-Supervised Learning (MAE + Fine-tuning)
This script runs in two phases: (1) MAE Pretraining -> (2) Segmentation Fine-tuning.
python train_problem3.py



## 📦 Generating Submission
To generate predictions for submission:
```
bash script/run_problem1.sh
```
(Example for Problem 1)
To generate predictions for submission using the ensemble of 5 trained models:
This script automatically downloads weights and runs inference
bash scripts/run_problem1.sh <PUBLIC_DIR> <PRIVATE_DIR>


# For Problem 1 (5-Fold Ensemble)
bash scripts/run_problem1.sh <PUBLIC_DIR> <PRIVATE_DIR>
# For Problem 2 (Nested-TransUNet)
bash scripts/run_problem2.sh <PUBLIC_DIR> <PRIVATE_DIR>
# For Problem 3 (MAE SSL)
bash scripts/run_problem3.sh <PUBLIC_DIR> <PRIVATE_DIR>


## 🧠 Model Architecture

### Problem 1: Simple-CNN UNet
We implemented a MedicalResUNet, which enhances the standard U-Net architecture by integrating Residual Blocks. This allows for deeper network training and better feature preservation.
Encoder: 4 stages of downsampling using MaxPool, each followed by Residual Blocks.
Bottleneck: A deep Residual Block to capture high-level semantic features.
Decoder: Uses Transpose Convolution for upsampling, concatenated with skip connections from the encoder to recover spatial details.
Output: A 1x1 convolution followed by a Sigmoid activation



### Problem 2: Trnasformer + UNet
We designed a HybridViTSegmenter that combines the strengths of UNet++ and Vision Transformers.
Backbone (Deep Stem): A custom ResNet-like CNN stem to extract local features and provide a strong inductive bias.
Transformer Bottleneck: Replaces the deepest CNN layer with Transformer Blocks to capture global long-range dependencies using Self-Attention.
Decoder (UNet++): Utilizes Dense Skip Connections to improve feature fusion and gradient flow.
Attention Gates: Integrated into skip connections to filter out background noise.
Deep Supervision: The model outputs predictions from 4 different depths, enforcing feature learning at all levels.





## Model Configuration

### Problem 1: Simple-CNN UNet
Problem 1: Simple-CNN UNetTo achieve high performance and stability, we utilized a 5-Fold Cross-Validation Ensemble strategy.Architecture: MedicalResUNet (Manual Implementation)Input Size: 256 x 256Epochs: 150 per foldBatch Size: 16Optimizer: AdamW (weight_decay=1e-4)Scheduler: CosineAnnealingLR (Decays from 3e-4 to 1e-6)Loss Function: Log-Cosh Dice LossRationale: Standard Dice Loss became unstable when scores exceeded 0.7. Log-Cosh smoothing provided a stable gradient landscape, allowing the model to converge to a higher optima without oscillation.Data Augmentation (Safe-Policy):Geometric: Random Horizontal/Vertical Flip, Moderate Affine (Rotation $\pm 10^\circ$, Scaling 0.85-1.15).Pixel-level: Gaussian Blur, Gaussian Noise.Color: Random Brightness, Contrast, and Gamma adjustments.

### Problem 2: Trnasformer + UNet
To overcome the difficulty of training ViT on small datasets from scratch, we introduced Deep Supervision and Gradient Accumulation.
Architecture: Nested-TransUNet (HybridViTSegmenter)
Configuration: embed_dim=512, depth=4, heads=8
Epochs: 200
Batch Size: 8 (Accumulated to effective batch size of 32)
Optimizer: AdamW (weight_decay=0.03)
Scheduler: CosineAnnealingLR (Decays from 3e-4 to 1e-6)
Loss Function: Deep Supervision Loss (Weighted sum of Log-Cosh Dice Loss from 4 decoder levels).


## 🔍 Self-Supervised Learning(Problem 3)
Method Design: Masked Autoencoder (MAE)
To address the limitation of training Transformers on a small dataset (~3000 images) without external pre-trained weights, we designed a Masked Image Reconstruction pretext task based on the MAE (Masked Autoencoder) philosophy.
Pretext Task (Masked Image Modeling): * We randomly mask 75% of the image patches (16x16).
The model is tasked with reconstructing the missing pixels from the remaining 25% visible patches.
Motivation: This high masking ratio forces the Transformer bottleneck to learn robust global semantic features and understand anatomical structures (shapes, continuity) rather than relying on local interpolation.
Dataset Expansion:
We utilized both the labeled Training Set and the unlabeled Public Test Set for the pretraining phase, effectively increasing the data scale for feature learning.
Training Strategy (Two-Stage):
Phase 1 (Pretraining): The Nested-TransUNet (with a 3-channel reconstruction head) is trained to minimize L1 Loss between the reconstructed image and the original image.
Phase 2 (Fine-tuning): We load the pre-trained Encoder and Decoder weights, replace the head with a 1-channel segmentation head, and fine-tune using Deep Supervision and Log-Cosh Dice Loss. A lower learning rate (1e-4) is used to preserve the learned representations.

## 📈 Learning Curve
Include the following plots:
- Learning rate (per step)
- Training loss (per step)
- Validation Dice Score (per epoch)
> [!TIP]
> You can download these logs and plots directly from Weights & Biases (wandb).
Problem 1: Training Dynamics
Below are the training loss (Log-Cosh Dice) and validation Dice scores across the 5 folds. The model consistently converged to a validation Dice score > 0.74.
Training Loss
![Training Loss Curve](assets/p1_loss.jpg)
Validation Dice Score
![Validation Dice Curve](assets/p1_dice.jpg)

Problem 2: Nested-TransUNet (From Scratch)
The training dynamics of the Hybrid ViT model trained from scratch with Deep Supervision.

| ![P2 Loss](assets/p2_loss.jpg) | ![P2 Dice](assets/p2_dice.jpg) |
| *Figure 2-A: Loss convergence with Deep Supervision.* | *Figure 2-B: Validation Dice showing gradual improvement.* |

Problem 3: Self-Supervised Learning (MAE Strategy)
We visualize both the pretraining phase (Reconstruction Task) and the fine-tuning phase (Segmentation Task).
#### Phase 1: MAE Pretraining (Reconstruction)
The model learns to reconstruct images from 25% visible patches. Lower L1 Loss indicates better understanding of global structures.
![P3 Recon Loss](assets/p3_recon_loss.jpg)
*Figure 3-A: MAE Reconstruction Loss decreasing during the pretext task.*

#### Phase 2: Segmentation Fine-tuning
After initializing with MAE weights, the model is fine-tuned for segmentation.
| ![P3 Finetune Loss](assets/p3_finetune_loss.jpg) | ![P3 Finetune Dice](assets/p3_finetune_dice.jpg) |




## 🧪 Experiment Results

### 1. Data Augmentation Ablation (Problem 1)
We investigated the impact of data augmentation intensity on model convergence. We observed that extremely strong geometric augmentations (e.g., 90-degree rotation) harmed feature learning on this small dataset, likely due to excessive distribution shift. A "Safe-Strong" policy yielded the best stability and performance.

| Data Augmentation Strategy | Description | Validation DSC (Avg) 
| Basic Augmentation | Random Flip only | ~0.70 |
| Safe-Strong Augmentation (Proposed) | Flip + Moderate Affine + Pixel-level Noise/Blur | 0.753 (Avg of 5 folds) |

### 2. Loss Function Ablation (Problem 1)
We compared the standard Dice Loss with our proposed Log-Cosh Dice Loss. The standard Dice Loss exhibited high volatility when validation scores exceeded 0.7, causing the optimizer to overshoot local minima. Log-Cosh smoothing provided a stable gradient landscape for fine-tuning.

| Loss Function | Observations | Validation DSC (Peak) |
| :--- | :--- | :--- |
| Standard Dice Loss | Unstable gradients near convergence, high variance | 0.7307 |
| Log-Cosh Dice Loss | Smooth convergence, stable improvement | 0.7665 |

### 3. Final Model Comparison & Leaderboard Performance
We evaluated three distinct approaches to address the medical image segmentation task. The results highlight the trade-off between Inductive Bias (CNN) and Model Capacity (Transformer) under limited data constraints.

| Problem | Model Architecture | Training Strategy | Validation DSC (Best) | Public Leaderboard DSC |
| Problem 1 | MedicalResUNet (CNN) | 5-Fold Ensemble + TTA| 0.7665 | 0.5344 |
| Problem 2 | Nested-TransUNet | Deep Supervision (Scratch) | 0.7405 | 0.4967 |
| Problem 3 | Nested-TransUNet | MAE SSL Pretraining | 08580 | 0.4605 |


## 💡 Observations & Expert Insights

### Problem 1: The Victory of Inductive Bias (Score: 0.5344)
Analysis: The CNN-based MedicalResUNet achieved the highest public leaderboard score (0.5344).
Reasoning:Medical datasets are typically small (~3000 images). CNNs possess a strong Inductive Bias(translation invariance and locality), allowing them to learn robust edge and texture features efficiently even with limited data.
Strategy Success:The 5-Fold Ensemble strategy successfully mitigated the variance inherent in single-model training, proving to be the most reliable method for this task.

### Problem 2: The Overfitting Trap of Transformers (Score: 0.4967)
Analysis: While the Nested-TransUNet achieved a respectable Validation Dice (0.7405), its Public Leaderboard score dropped significantly to 0.4967
Reasoning: This large gap indicates severe Overfitting. Vision Transformers (ViT) lack the inductive bias of CNNs and require massive datasets (e.g., ImageNet-21k) to learn generalizable features. Training a complex ViT from scratch on a small medical dataset leads to the model "memorizing" the training data but failing to generalize to the domain-shifted test set.
Conclusion:Advanced architectures do not guarantee better results without sufficient data or pretraining.

### Problem 3: Bridging the Gap with SSL (Proposed Solution)
Motivation: To solve the overfitting issue seen in Problem 2, we introduced Self-Supervised Learning (SSL) using a Masked Autoencoder (MAE) approach.
Methodology:
    1.  Pretext Task: We mask 75% of the image patches and force the Nested-TransUNet to reconstruct the missing anatomical structures.
    2.  Effect: This forces the Transformer to learn robust global semantic features and "understanding" of the image structure *before* seeing any segmentation labels.
    3.  Hypothesis: By initializing the segmentation model with these "structure-aware" weights, we aim to recover the generalization capability that the from-scratch ViT lacked, potentially surpassing the CNN baseline.

📚 Notes
- Do not upload model checkpoints or data to GitHub repository.
- Ensure your results are reproducible by fixing random seeds and clearly specifying configurations.
- Figures, Tables, and experiment descriptions should be self-contained and interpretable without external references.
- Please maintain clarity and organization throughout your report and code structure.
