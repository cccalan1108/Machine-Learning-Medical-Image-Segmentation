set -e

PUBLIC_DIR=$1
PRIVATE_DIR=$2


mkdir -p checkpoints

echo "Running Inference for Problem 1 (5-Fold Ensemble)"

# 1. 下載權重 (請填入雲端連結)
# 檢查是否已存在，不存在才下載
if [ ! -f "checkpoints/p1/best_model_fold_0.pth" ]; then
    echo "Downloading Fold 0 weights..."
    # wget -O checkpoints/best_model_fold_0.pth "請填入_Fold0_的下載連結"
fi

if [ ! -f "checkpoints/p1/best_model_fold_1.pth" ]; then
    echo "Downloading Fold 1 weights..."
    # wget -O checkpoints/best_model_fold_1.pth "請填入_Fold1_的下載連結"
fi

if [ ! -f "checkpoints/p1/best_model_fold_2.pth" ]; then
    echo "Downloading Fold 2 weights..."
    # wget -O checkpoints/best_model_fold_2.pth "請填入_Fold2_的下載連結"
fi

if [ ! -f "checkpoints/p1/best_model_fold_3.pth" ]; then
    echo "Downloading Fold 3 weights..."
    # wget -O checkpoints/best_model_fold_3.pth "請填入_Fold3_的下載連結"
fi

if [ ! -f "checkpoints/p1/best_model_fold_4.pth" ]; then
    echo "Downloading Fold 4 weights..."
    # wget -O checkpoints/best_model_fold_4.pth "請填入_Fold4_的下載連結"
fi

python inference_problem1.py "$PUBLIC_DIR" "$PRIVATE_DIR"

echo "Problem 1 Inference Completed"