set -e

PUBLIC_DIR=$1
PRIVATE_DIR=$2

mkdir -p checkpoints

echo "Running Inference for Problem 3 (MAE SSL)"

if [ ! -f "checkpoints/p3/best_model_p3.pth" ]; then
    echo "Downloading Problem 3 weights..."
    # wget -O checkpoints/best_model_p3.pth "請填入_P3_權重的下載連結"
fi

python inference_problem3.py "$PUBLIC_DIR" "$PRIVATE_DIR"

echo "Problem 3 Inference Completed"