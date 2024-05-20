export CUDA_VISIBLE_DEVICES="0,1,2,5,6"
python -m torch.distributed.launch \
    --nproc_per_node=${NPROC_PER_NODE:-5} \
    --nnodes=${WORLD_SIZE:-1} \
    --node_rank=${RANK:-0} \
    --master_addr=${MASTER_ADDR:-127.0.0.1} \
    --master_port=${MASTER_PORT:-12345} \
    ../eval.py \
    --model_name minicpm \
    --model_path /home/zhanghaoye/MiniCPMV_checkpoints_zhy/DPO_exp/minicpmv_llama3_multilingual/minicpmv_DPO-minicpmv_llama3_multilingual_1iter_greedy_sr4000img_bs1_gradacc4_beta0.3_lr5e-7_fp32-minicpmv_llama3_multilingual_1iter_greedy_sr4000img-1/checkpoints/checkpoint-280 \
    --generate_method interleave \
    --eval_textVQA \
    --eval_docVQATest \
    --answer_path ../answers_vqa/answers-0519 \
    --batchsize 1