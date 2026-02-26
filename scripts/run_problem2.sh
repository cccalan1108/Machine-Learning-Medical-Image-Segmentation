set -e

PUBLIC_DIR=$1
PRIVATE_DIR=$2
mkdir -p checkpoints
echo "Running Inference for Problem 2 (Nested-TransUNet)"

# 1. 下載權重
# 注意：P2 的 inference script 預設讀取 checkpoints/best_model.pth
if [ ! -f "checkpoints/p2/best_model.pth" ]; then
    echo "Downloading Problem 2 weights..."
    # wget -O checkpoints/best_model.pth "請填入_P2_權重的下載連結"
fi

python inference_problem2.py "$PUBLIC_DIR" "$PRIVATE_DIR"

echo "Problem 2 Inference Completed"