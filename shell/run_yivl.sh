export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m torch.distributed.launch \
    --nproc_per_node=${NPROC_PER_NODE:-8} \
    --nnodes=${WORLD_SIZE:-1} \
    --node_rank=${RANK:-0} \
    --master_addr=${MASTER_ADDR:-127.0.0.1} \
    --master_port=${MASTER_PORT:-12345} \
    ../eval.py \
    --model_name yivl \
    --model_path /data/public/multimodal/multimodal_model_ckpts/01-ai/Yi-VL-34B \
    --eval_llavabench_multilingual \
    --answer_path ../answers_vqa/answers-0519
    --batchsize 1