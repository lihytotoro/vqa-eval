export CUDA_VISIBLE_DEVICES="4,5"
python -m torch.distributed.launch \
    --nproc_per_node=${NPROC_PER_NODE:-2} \
    --nnodes=${WORLD_SIZE:-1} \
    --node_rank=${RANK:-0} \
    --master_addr=${MASTER_ADDR:-127.0.0.1} \
    --master_port=${MASTER_PORT:-12347} \
    ../eval.py \
    --model_name CodeLlama-7b \
    --model_base_dir /data/public/multimodal/lihaoyu/szx/training_output \
    --template_name llama2 \
    --load_in_4bit \
    --eval_d4j \
    --data_dir /data/public/multimodal/lihaoyu/szx/datasets/d4j-processed/processed \
    --test_file defects4j_all_single_func_repairllama_wo_initial_prompt.jsonl \
    --max_input_len 1024 \
    --max_output_len 256 \
    --do_sample \
    --do_beam \
    --num_beams 10 \
    --top_p 0.95 \
    --top_k 50 \
    --temperature 0.8 \
    --repetition_penalty 1.0 \
    --request_num 10 \
    --output_base_dir /data/public/multimodal/lihaoyu/szx/inference_output \
    --method old-method \
    --task_type direct-apr \
    --sft_type lora \
    --sft_epoch 1
