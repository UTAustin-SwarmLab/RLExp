# RLExp

## Verl
pip install -e verl

## Prepare dataset
python3 src/data/math_meta.py --local_save_dir data/math2
python3 src/data/gsm8k_meta.py --local_save_dir data/gsm8k

#% Run training
VLLM_USE_V1=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh src/scripts/train_grpo.sh math2