export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"
python -m torch.distributed.launch \
    --nproc_per_node=${NPROC_PER_NODE:-7} \
    --nnodes=${WORLD_SIZE:-1} \
    --node_rank=${RANK:-0} \
    --master_addr=${MASTER_ADDR:-127.0.0.1} \
    --master_port=${MASTER_PORT:-12345} \
    ../eval.py \
    --model_name minigemini \
    --model_path /data/public/multimodal/multimodal_model_ckpts/YanweiLi/MGM-2B \
    --eval_objhal \
    --objhal_ann_path /data/public/multimodal/multimodal_data/dpo/eval/CHAIR/chair_instruction2_img_diversify_8_noemo.jsonl \
    --answer_path ../answers_vqa/answers-0516 \
    --batchsize 1