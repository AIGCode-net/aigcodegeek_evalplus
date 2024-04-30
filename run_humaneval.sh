mkdir -p results/humaneval

VLLM_N_GPUS=1 python generate.py \
    --model_series aigcode \
    --release_tag v1 \
    --dataset humaneval \
    --root generations \
    --greedy True 

evalplus.evaluate \
      --dataset humaneval \
      --samples generations/humaneval/aigcode--AIGCodeGeek-DS-6.7B_temp_0.0 > results/humaneval/aigcodev1_temp_0.0.txt