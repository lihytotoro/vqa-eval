export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"
python -m torch.distributed.launch \
    --nproc_per_node=${NPROC_PER_NODE:-7} \
    --nnodes=${WORLD_SIZE:-1} \
    --node_rank=${RANK:-0} \
    --master_addr=${MASTER_ADDR:-127.0.0.1} \
    --master_port=${MASTER_PORT:-12345} \
    eval.py \
    --model_name minicpm \
    --model_path /home/cuijunbo/0429_model/minicpmv_llama3_9k5/minicpmv_DPO-minicpmv_llama3_9k5_base_filtwrong_bs1_gradacc4_beta0.3_lr5e-7_fp32_correctconfig-minicpmv_llama3_9k5_base_filterwrong_sr4000img-1/checkpoints/checkpoint-468 \
    --objhal_ann_path /data/public/multimodal/multimodal_data/dpo/eval/CHAIR/chair_instruction2_img_diversify_8_noemo.jsonl \
    --generate_method interleave \
    --eval_textVQA \
    --eval_docVQA \
    --answer_path ./answers-0510-1 \
    --batchsize 1