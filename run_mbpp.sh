mkdir -p results/mbpp

VLLM_N_GPUS=1 python generate.py \
    --model_series aigcode \
    --release_tag v1 \
    --dataset mbpp \
    --root generations \
    --greedy True 

evalplus.evaluate \
      --dataset mbpp \
      --samples generations/mbpp/aigcode--AIGCodeGeek-DS-6.7B_temp_0.0 > results/mbpp/aigcodev1_temp_0.0.txt