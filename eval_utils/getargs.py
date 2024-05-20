import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    parser.add_argument('--local-rank', type=int, default=0, help='Local rank for distributed training')
    #OCR datasets
    parser.add_argument("--ocr_dir_path", type=str, default="./data")
    parser.add_argument("--ocr_dataset_name", type=str, default="IIIT5K svt IC13_857 IC15_1811 svtp ct80 cocotext ctw totaltext HOST WOST WordArt")

    #textVQA
    parser.add_argument("--textVQA_image_dir_path", type=str, default="/data/public/multimodal/multimodal_data/OCR_eval/TextVQA/train_images")
    parser.add_argument("--textVQA_ann_path", type=str, default="/data/public/multimodal/multimodal_data/OCR_eval/TextVQA/TextVQA_0.5.1_val.json")

    #docVQA
    parser.add_argument("--docVQA_image_dir_path", type=str, default="/data/public/multimodal/multimodal_data/OCR_eval/DocVQA")
    parser.add_argument("--docVQA_ann_path", type=str, default="/data/public/multimodal/multimodal_data/OCR_eval/DocVQA/val_v1.0_withQT.json")

    #ocrVQA
    parser.add_argument("--ocrVQA_image_dir_path", type=str, default="./data/ocrvqa/images")
    parser.add_argument("--ocrVQA_ann_path", type=str, default="./data/ocrvqa/ocrvqa_val.jsonl")

    #STVQA
    parser.add_argument("--STVQA_image_dir_path", type=str, default="./data/STVQA")
    parser.add_argument("--STVQA_ann_path", type=str, default="./data/STVQA/train_task_3.json")

    parser.add_argument("--FUNSD_dir_path", type=str, default="./data/FUNSD/testing_data/annotations")
    
    # VQAv2
    parser.add_argument("--VQAv2_image_dir_path", type=str, default="/data/public/multimodal/multimodal_data/coco/coco2017")
    parser.add_argument("--VQAv2_annotation_path", type=str, default="./data/VQAv2/v2_mscoco_val2014_annotations.json")
    parser.add_argument("--VQAv2_question_path", type=str, default="./data/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json")

    # chair dataset
    parser.add_argument("--objhal_ann_path", type=str, default="", help="path to ann of the chair dataset(objhal).")

    #result_path
    parser.add_argument("--answer_path", type=str, default="./answers-new")

    #eval
    parser.add_argument(
        "--eval_IDLOCR",
        action="store_true",
        default=False,
        help="Whether to evaluate on IDLOCR"
    )
    parser.add_argument(
        "--eval_objhal",
        action="store_true",
        default=False,
        help="Whether to evaluate on C4WEB."
    )
    parser.add_argument(
        "--eval_C4WEB",
        action="store_true",
        default=False,
        help="Whether to evaluate on C4WEB."
    )
    parser.add_argument(
        '--eval_WebSRC',
        action="store_true",
        default=False,
        help="Whether to evaluate on WebSRC."
    )
    parser.add_argument(
        '--eval_VisualMRC',
        action="store_true",
        default=False,
        help="Whether to evaluate on VisualMRC."
    )
    parser.add_argument(
        '--eval_HY',
        action="store_true",
        default=False,
        help="Whether to evaluate on HY."
    )
    parser.add_argument(
        "--eval_YFVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on YFVQA."
    )
    parser.add_argument(
        "--eval_VQAv2",
        action="store_true",
        default=False,
        help="Whether to evaluate on VQAv2."
    )
    parser.add_argument(
        "--eval_textVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on textVQA."
    )
    parser.add_argument(
        "--eval_docVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on docVQA."
    )
    parser.add_argument(
        "--eval_docVQATest",
        action="store_true",
        default=False,
        help="Whether to evaluate on docVQA."
    )
    parser.add_argument(
        "--eval_ocrVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on ocrVQA."
    )    
    parser.add_argument(
        "--eval_keepVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on ocrVQA."
    )
    parser.add_argument(
        "--eval_STVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on STVQA."
    )

    parser.add_argument(
        "--eval_FUNSD",
        action="store_true",
        default=False,
        help="Whether to evaluate on FUNSD."
    )
    
    # 0329 new: eval_mathvista_mmvet
    parser.add_argument(
        "--eval_mathvista_mmvet", 
        action="store_true", 
        default=False, 
        help="Whether to evaluate on both MathVista and MMVet."
    )
    
    parser.add_argument(
        "--eval_which_in_mathvista_mmvet",
        type=str,
        default="MathVista",
        help="Which Dataset should be tested while eval_mathvista_mmvet is true.",
    )
    
    # 0411 new: eval_grounding
    parser.add_argument(
        "--eval_grounding",
        action="store_true",
        default=False,
        help="Whether to eval on Grounding dataset(WebSimulator)."
    )
    
    parser.add_argument(
        "--grounding_dataset_dir",
        type=str,
        default="",
        help="dir to the grounding dataset."
    )
    
    parser.add_argument(
        "--grounding_dataset_dir_type",
        type=str,
        default="single",
        help="directly to the images or layers of dir."
    )
    
    parser.add_argument(
        "--day_subdir_path",
        type=str,
        default="",
        help="date of today to discriminate different experiments."
    )
    
    # 0423 new: eval new ocr sft dataser:
    parser.add_argument(
        "--eval_ocrsft", 
        action="store_true",
        default=False,
        help="Whether to eval on new ocrsft dataset."
    )
    
    parser.add_argument(
        "--ocrsft_dataset_path",
        type=str,
        default="",
        help="path to ocrsft dataset."
    )
    
    # 0430 new: eval ocr code dataset (50 samples)
    parser.add_argument(
        "--eval_ocrcode", 
        action="store_true",
        default=False,
        help="Whether to eval on new ocrcode dataset."
    )
    
    parser.add_argument(
        "--ocrcode_dataset_path",
        type=str,
        default="",
        help="path to ocrcode dataset."
    )
    
    parser.add_argument(
        "--eval_case",
        action="store_true",
        default=False,
        help="Whether to evaluate on case."
    )
    parser.add_argument(
        "--eval_ocr",
        action="store_true",
        default=False,
        help="Whether to evaluate on ocr."
    )
    
    # 0519 new: eval llava_bench mulitilingual
    parser.add_argument(
        "--eval_llavabench_multilingual",
        action="store_true",
        default=False,
        help="whether to evaluate on llava bench multilingual dataset(4 * 6 = 24)."
    )
    
    # 0513 new: eval_d4j
    parser.add_argument(
        "--eval_d4j",
        action="store_true",
        default=False,
        help="Whether to evaluate on d4j dataset(single func subset)."
    )
    # 0515 new:
    parser.add_argument(
        "--eval_cweinf",
        action="store_true",
        default=False,
        help="Whether to evaluate on cweinf task."
    )
    parser.add_argument("--cwe_dataset_name", type=str, default="java-juliet", help="")
    
    parser.add_argument(
        "--eval_all",
        action="store_true",
        default=False,
        help="Whether to evaluate all datasets"
    )

    # parser.add_argument("--Pix2Struct_large_path", type=str, default="/home/zhangli/llama_models/llama/llama-7b")
    # parser.add_argument('--font_path', type=str, default="/home/cuijunbo/model/models--ybelkada--fonts/snapshots/7f29c3755a0de4c552c4b474cef01d10eb3d5e8b/Arial.TTF", help='font_path of Pix2Struct')
    parser.add_argument("--model_name", type=str, default="BLIP2")
    # 这个参数一般用于 codellama（szx）
    parser.add_argument("--model_base_dir", type=str, default="", help="")
    # 这个任务一般用于原始的 vqa test 部分
    parser.add_argument("--model_path", type=str, default="/home/cuijunbo/0414_coding/multimodal_rhapsody/base_models/MiniCPM-V-2")

    parser.add_argument("--generate_method", type=str, default="", help="generate with interleave or not.")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument('--batchsize', type=int, default=1, help='Batch size for processing.')

    parser.add_argument("--ckpt", type=str, default="/home/zhangli/code/open_flamingo/checkpoint/checkpoint.pt")

    parser.add_argument("--sample_start_idx", type=int, default=0, help="")
    parser.add_argument("--sample_end_idx", type=int, default=0, help="")

    # 0513 new: related to apr
    parser.add_argument("--template_name", type=str, default="", help="templated related to the model.")
    parser.add_argument("--load_in_4bit", action="store_true", default=False, help="Whether to load in 4bit while doing inference.")
    parser.add_argument("--load_in_8bit", action="store_true", default=False, help="Whether to load in 8bit while doing inference.")
    
    # 用于测试的原始数据的位置
    # 目前的路径：/data/public/multimodal/lihaoyu/szx/datasets/d4j-processed/processed/defects4j_all_single_func_repairllama_wo_initial_prompt.jsonl
    # 在 chat 模式下，暂时不起作用
    parser.add_argument("--data_dir", type=str, default="", help="")
    parser.add_argument("--test_file", type=str, default="", help="name of the file which contains the samples to be tested.")
    
    # 输出文件的名称
    # 在 APR 任务中需要
    parser.add_argument("--output_base_dir", type=str, default="", help="")
    # parser.add_argument("--output_file", type=str, default="", help="name of the inference output file.")
    
    # 下面的参数负责控制输入输出的 token 数目
    # Corresponds to the length of the input prompt + max_new_tokens. Its effect is overridden by max_new_tokens, if also set.
    parser.add_argument("--max_input_len", type=int, default=1024, help="max len of input tokens.")
    parser.add_argument("--max_output_len", type=int, default=256, help="max len of output tokens.")
    
    # 下面的参数负责控制生成的参数
    parser.add_argument("--do_sample", action="store_true", default=False, help="Whether to use sampling.")
    parser.add_argument("--do_beam", action="store_true", default=False, help="Whether to only use beam search.")
    parser.add_argument("--do_topp", action="store_true", default=False, help="Whether to use top-p sampling.")
    parser.add_argument("--do_topk", action="store_true", default=False, help="Whether to use top-k sampling.")
    parser.add_argument("--do_temp", action="store_true", default=False, help="Whether to use temperature sampling.")
    parser.add_argument("--num_beams", type=int, default=1, help="number of beams in beam search.")
    parser.add_argument("--top_p", type=float, default=0.9, help="")
    parser.add_argument("--top_k", type=int, default=50, help="")
    parser.add_argument("--temperature", type=float, default=0.35, help="")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="")
    
    # 这里是问题关键所在！之前直接将这里设置为 10，显存占用很大，但是实际上应该是 num_beams 设置的过大导致的
    parser.add_argument("--request_num", type=int, default=1, help="Number of requests.")

    # 以下参数用于选择运行实验的设置（使用的模型、输出的目录）
    parser.add_argument("--method", type=str, default="old-method", help="old or new thought.")
    parser.add_argument("--task_type", type=str, default="direct-apr", help="which task to run.")
    parser.add_argument("--sft_type", type=str, default="qlora", help="full, lora, or qlora while sft.")
    parser.add_argument("--sft_epoch", type=int, default=1, help="how many epochs was trained during sft.")

    args = parser.parse_args()
    return args