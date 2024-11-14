export CALVIN_ROOT="/data/calvin"
export CUDA_VISIBLE_DEVICES=0,3
export PYTHONPATH="/data/vidman":$PYTHONPATH
accelerate launch scripts/evaluate.py configs/eval_config.py