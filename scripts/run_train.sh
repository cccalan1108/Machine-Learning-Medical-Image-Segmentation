#!/bin/bash
set -e


#預先建立分類資料夾 
mkdir -p checkpoints/p1
mkdir -p checkpoints/p2
mkdir -p checkpoints/p3


echo "[Problem 1] Starting Training (5-Folds)"



for fold in 0 1 2 3 4
do
    echo "Running Fold $fold..."
    python train_problem1.py --fold $fold
    
    if [ -f "checkpoints/best_model_fold_$fold.pth" ]; then
        mv "checkpoints/best_model_fold_$fold.pth" "checkpoints/p1/best_model_fold_$fold.pth"
        echo "Moved Fold $fold weights to checkpoints/p1/"
    fi
done

echo "[Problem 2] Starting Training: Nested-TransUNet"

python train_problem2.py

if [ -f "checkpoints/best_model.pth" ]; then
    mv "checkpoints/best_model.pth" "checkpoints/p2/best_model.pth"
    echo "Saved P2 weights to checkpoints/p2/best_model.pth"
fi


echo "[Problem 3] Starting Training: MAE SSL + Fine-tuning"

python train_problem3.py


if [ -f "checkpoints/best_model_p3.pth" ]; then
    mv "checkpoints/best_model_p3.pth" "checkpoints/p3/best_model_p3.pth"
    echo "Saved P3 weights to checkpoints/p3/best_model_p3.pth"
elif [ -f "checkpoints/best_model.pth" ]; then
    mv "checkpoints/best_model.pth" "checkpoints/p3/best_model_p3.pth"
    echo "Saved P3 weights to checkpoints/p3/best_model_p3.pth"
fi

