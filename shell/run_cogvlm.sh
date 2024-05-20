export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m torch.distributed.launch \
    --nproc_per_node=${NPROC_PER_NODE:-8} \
    --nnodes=${WORLD_SIZE:-1} \
    --node_rank=${RANK:-0} \
    --master_addr=${MASTER_ADDR:-127.0.0.1} \
    --master_port=${MASTER_PORT:-12348} \
    eval.py \
    --model_name cogvlm \
    --eval_objhal \
    --answer_path ./answers-new \
    --batchsize 1