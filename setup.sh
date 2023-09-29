mkdir /workspace/cache
export TRANSFORMERS_CACHE=/workspace/cache/
pip install -r requirements.txt
apt update && apt install htop
wandb login
huggingface-cli login