export CUDA_VISIBLE_DEVICES="3,5,6,7"
python -m torch.distributed.launch \
    --nproc_per_node=${NPROC_PER_NODE:-4} \
    --nnodes=${WORLD_SIZE:-1} \
    --node_rank=${RANK:-0} \
    --master_addr=${MASTER_ADDR:-127.0.0.1} \
    --master_port=${MASTER_PORT:-12346} \
    ../eval.py \
    --model_name Qwen1.5-7B-Chat \
    --model_base_dir /data/public/multimodal/lihaoyu/szx/training_output \
    --template_name qwen \
    --eval_cweinf \
    --cwe_dataset_name Java-Juliet \
    --data_dir /data/public/multimodal/lihaoyu/szx/datasets/java-juliet/src/parsed_dataset/jsonl/cwe-inference \
    --test_file finetuning_data_maxlen=1024_modeltype=qwen_usertype=qlora_dataset=cwe-inference_trainratio=0.9_split=test.jsonl \
    --max_input_len 1024 \
    --max_output_len 256 \
    --do_sample \
    --do_beam \
    --num_beams 1 \
    --top_p 0.95 \
    --top_k 50 \
    --temperature 0.8 \
    --repetition_penalty 1.0 \
    --request_num 1 \
    --output_base_dir /data/public/multimodal/lihaoyu/szx/inference_output \
    --method old-method \
    --task_type cwe-inference \
    --sft_type full \
    --sft_epoch 1
