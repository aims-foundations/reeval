python 6_compute_response_matrix.py \
    --dataset air-bench/air_bench_2024 \
    --question_generator stair-lab/reeval_Meta-Llama-3.1-8B-Instruct \
    --max_workers 2 \
    --force_run


python 7_calibrate_evaluate.py