export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
python -m torch.distributed.launch \
    --nproc_per_node=${NPROC_PER_NODE:-8} \
    --nnodes=${WORLD_SIZE:-1} \
    --node_rank=${RANK:-0} \
    --master_addr=${MASTER_ADDR:-127.0.0.1} \
    --master_port=${MASTER_PORT:-12348} \
    eval.py \
    --model_name minicpm \
    --model_path /home/cuijunbo/0429_model/minicpmv_llama3_9k5/minicpmv_DPO-minicpmv_llama3_9k5_base_filtwrong_bs1_gradacc4_beta0.3_lr5e-7_fp32_correctconfig-minicpmv_llama3_9k5_base_filterwrong_sr4000img-1/checkpoints/checkpoint-468
    --eval_ocrcode \
    --ocrcode_dataset_path /home/cuijunbo/LMUData/images/MME \
    --answer_path ./answers-0430 \
    --batchsize 1