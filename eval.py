import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.join(script_dir, '..'))
# sys.path.append("/home/lihaoyu/code/0410/MiniGemini")
sys.path.append("/home/lihaoyu/code/0516/MGM")

import datetime
import json
import os
from functools import partial

import torch

from datasets.kie_dataset import FUNSDDataset
from datasets.ocr_dataset import C4WEBDataset, IDLOCRDataset
from datasets.vqa_dataset import (CaseDataset, CasematDataset, ESTVQADataset,
                                  STVQADataset, VisualMRCDataset, VQAv2Dataset,
                                  WebSRCDataset, chairDataset, docVQADataset,
                                  docVQATESTDataset, ocrVQADataset,
                                  textVQADataset, GroundingDataset, OCRsftDataset, OCRcodeDataset, LlavaBenchMultilingualDataset)
# 0513 new
# 初期，我们先引入 d4j 的数据集类
from datasets.apr_dataset import d4jDataset, cweinfDataset


print(torch.__version__)

import numpy as np

from eval_utils.getargs import parse_args
# 这里引入了 evaluate_VAQ 等函数
from eval_utils.vqa_evaluate import *


def get_model(args):
    if args.model_name=='':
        pass
    elif 'pix2struct' in args.model_name.lower():
        from models.Pix2Struct.Pix2Struct import Pix2Struct
        
        model_path = args.ckpt
            
        model = Pix2Struct(model_path=model_path,device=args.device,font_path=args.font_path)  
    elif "minicpm" in args.model_name.lower():
        from models.MiniCPM.minicpmv import MiniCPM_V
        # 初始化路径
        model_path = args.model_path
        model = MiniCPM_V(model_path=model_path, device=args.device)
    elif 'Qwen_VL' in args.model_name:
        from models.QwenVL.Qwen_VL import QwenVL

        model_path = "/data/public/multimodal/multimodal_model_ckpts/Qwen_VL/Qwen-VL"
        model = QwenVL(model_path, args.device)
    elif "deepseek" in args.model_name.lower():
        from models.DeepSeek.deepseek import DeepSeekVLChat

        model_path = ""
        model = DeepSeekVLChat(model_path=model_path)
    elif "cogvlm" in args.model_name.lower():
        from models.CogVLM.cogvlm import CogVlm
        model = CogVlm()
    elif "yivl" in args.model_name.lower():
        from models.YiVL.yivl import YiVL
        model_path = args.model_path
        root_path = "/home/hongyixin/vqa_eval/models/YiVL/Yi"
        model = YiVL(model_path=model_path, root=root_path)
    # 0410 new: minigemini
    elif "minigemini" in args.model_name.lower():
        model_path = args.model_path
        from models.MiniGemini.minigemini import MiniGemini
        model = MiniGemini(model_path=model_path, device=args.device)
        
    # 0513 new: codellama
    elif "codellama" in args.model_name.lower():
        from models.CodeLLaMA.codellama import CodeLLaMA
        model = CodeLLaMA(args=args, device=args.device)
        
    # 0515 new: qwen
    elif "qwen" in args.model_name.lower():
        from models.Qwen.qwen import Qwen
        model = Qwen(args=args, device=args.device)
    
    return model

def main(args):
    np.random.seed(0)
    max_sample_num = None
    # max_sample_num = 500
    
    # args.max_sample_num = max_sample_num

    torch.distributed.init_process_group(
        backend='nccl',
        world_size=int(os.getenv('WORLD_SIZE', '1')),
        rank=int(os.getenv('RANK', '0')),
    )
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
    print(f'Init Rank-{torch.distributed.get_rank()}')
    if torch.distributed.is_initialized():
        args.device = torch.device(f"cuda:{torch.cuda.current_device()}")

    # Using torchrun
    # torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))

    # # print(f'Init Rank-{torch.distributed.get_rank()}')
    # if torch.distributed.is_initialized():
    #     args.device = torch.device(f"cuda:{torch.cuda.current_device()}")
    model = get_model(args)
    
    result = {}
    time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if args.eval_objhal:
        target_dataset = "objhal"
        
        dataset = chairDataset(ann_path=args.objhal_ann_path)
        
        acc = evaluate_VQA(model, dataset, args.model_name, 'objhal', time, batch_size=args.batchsize, generate_method=args.generate_method, answer_path=args.answer_path)
        result['objhal'] = acc
    if args.eval_case:
        # dataset = CaseDataset()
        dataset = CasematDataset()
        acc = evaluate_VQA(model, dataset, args.model_name, 'VisualMRC', time, batch_size=args.batchsize)
        result['case'] = acc
    if args.eval_WebSRC:
        dataset = WebSRCDataset()
        if max_sample_num is not None:
            # random sample
            dataset = torch.utils.data.Subset(dataset, torch.randperm(len(dataset)).tolist()[:max_sample_num])
        acc = evaluate_VQA(model, dataset, args.model_name, 'WebSRC', time, batch_size=args.batchsize)
        result['WebSRC'] = acc
    if args.eval_VisualMRC:
        dataset = VisualMRCDataset()
        # max_sample_num = 10
        if max_sample_num is not None:
            dataset = torch.utils.data.Subset(dataset, torch.randperm(len(dataset)).tolist()[:max_sample_num])
        acc = evaluate_VQA(model, dataset, args.model_name, 'VisualMRC', time, batch_size=args.batchsize)
        result['VisualMRC'] = acc

    if args.eval_textVQA or args.eval_all:
        dataset = textVQADataset(
            args.textVQA_image_dir_path, 
            args.textVQA_ann_path)
        if max_sample_num is not None:
            dataset = torch.utils.data.Subset(dataset, range(max_sample_num))
        
        # 看一下每个进程拿到的编号
        # 0429：
        # dataset = torch.utils.data.Subset(dataset, range(args.sample_start_idx, args.sample_end_idx))
            
        acc = evaluate_VQA(model, dataset, args.model_name, 'textVQA', time, \
                batch_size=args.batchsize, generate_method=args.generate_method, answer_path=args.answer_path, day_subdir_path=args.day_subdir_path, save_in_progress=True)
        result['textVQA'] = acc
        
    if args.eval_VQAv2 or args.eval_all:
        dataset = VQAv2Dataset(
            args.VQAv2_image_dir_path, 
            args.VQAv2_annotation_path,
            args.VQAv2_question_path, 
            )
        if max_sample_num is not None:
            dataset = torch.utils.data.Subset(dataset, range(max_sample_num))
        acc = evaluate_VQA(model, dataset, args.model_name, 'VQAv2', time, batch_size=args.batchsize)
        result['VQAv2'] = acc

    if args.eval_docVQA or args.eval_all:
        dataset = docVQADataset(args.docVQA_image_dir_path, args.docVQA_ann_path)
        if max_sample_num is not None:
            dataset = torch.utils.data.Subset(dataset, range(max_sample_num))
        acc = evaluate_VQA(model, dataset, args.model_name, 'docVQA', time, batch_size=args.batchsize, generate_method=args.generate_method, answer_path=args.answer_path)
        result['docVQA'] = acc
        
    # 0516 new    
    if args.eval_docVQATest or args.eval_all:
        target_dataset = "docVQATest"
        
        dataset = docVQATESTDataset('/data/public/multimodal/multimodal_data/OCR_eval/DocVQA', "/data/public/multimodal/multimodal_data/OCR_eval/DocVQA/test_v1.0.json")
        
        if max_sample_num is not None:
            dataset = torch.utils.data.Subset(dataset, range(max_sample_num))
        
        acc = evaluate_VQA(model, dataset, args.model_name, target_dataset, time, batch_size=args.batchsize, generate_method=args.generate_method, answer_path=args.answer_path)
        result['docVQATest'] = acc
    if args.eval_ocrVQA or args.eval_all:
        dataset = ocrVQADataset(args.ocrVQA_image_dir_path, args.ocrVQA_ann_path)
        if max_sample_num is not None:
            # dataset = torch.utils.data.Subset(dataset, torch.randperm(len(dataset)).tolist()[:max_sample_num])
            dataset = torch.utils.data.Subset(dataset, range(max_sample_num))
        acc = evaluate_VQA(model, dataset, args.model_name, 'ocrVQA', time, batch_size=args.batchsize)
        result['ocrVQA'] = acc

    if args.eval_STVQA or args.eval_all:
        dataset = STVQADataset(args.STVQA_image_dir_path, args.STVQA_ann_path)
        if max_sample_num is not None:
            dataset = torch.utils.data.Subset(dataset, range(max_sample_num))
        acc = evaluate_VQA(model, dataset, args.model_name, 'STVQA', time, batch_size=args.batchsize)
        result['STVQA'] = acc
    if args.eval_FUNSD or args.eval_all:
        dataset = FUNSDDataset(args.FUNSD_dir_path)
        if max_sample_num is not None:
            dataset = torch.utils.data.Subset(dataset, range(max_sample_num))

        acc = evaluate_VQA(model, dataset, args.model_name, 'FUNSD', time,batch_size=args.batchsize)
        result['FUNSD'] = acc
    
    if args.eval_C4WEB or args.eval_all:
        dataset = C4WEBDataset()
        if max_sample_num is not None:
            dataset = torch.utils.data.Subset(dataset, range(max_sample_num))
        acc = evaluate_VQA(model, dataset, args.model_name, 'C4WEB', time, batch_size=args.batchsize)
        result['C4WEB'] = acc
    if args.eval_IDLOCR or args.eval_all:
        dataset = IDLOCRDataset()
        if max_sample_num is not None:
            dataset = torch.utils.data.Subset(dataset, range(max_sample_num))
        acc = evaluate_VQA(model, dataset, args.model_name, 'IDLOCR', time, batch_size=args.batchsize)
        result['IDLOCR'] = acc
        
    # 0329 new: eval_mathvista_mmvet
    if args.eval_mathvista_mmvet or args.eval_all:
        # 仿照 casedataset 构造数据集，进行测试？
        target_dataset = "MathVista"
        
        if target_dataset == "MathVista":
            target_sample_path = "/home/lihaoyu/code/0328/output/mathvista_samples.jsonl"
        elif target_dataset == "MMVet":
            target_sample_path = "/home/lihaoyu/code/0328/output/mmvet_samples.jsonl"
        else:
            raise Exception("MathVista or MMVet required!")
        
        dataset = MathVistaMMVetDataset(target_samples_path=target_sample_path)
        
        # 注意，这里因为这个数据集没有 answer（gt），因此不应该返回 acc，我们需要的应该是真实回复
        res = evaluate_VQA(model, dataset, args.model_name, target_dataset, time, batch_size=args.batchsize)
        
        assert res < 0
        
        result[target_dataset] = res
        
    # 0411 new: grounding with minicpm-v
    if args.eval_grounding or args.eval_all:
        target_dataset = "Grounding"
        
        # 0414: 这里换用 cjb 提供的 1729 张 png 构成的 “测试集” 试一下
        # dataset = GroundingDataset(bbox_dir="/home/lihaoyu/code/0411/valid_samples_bbox_all_red", bbox_dir_type="layers")
        dataset = GroundingDataset(bbox_dir=args.grounding_dataset_dir, bbox_dir_type=args.grounding_dataset_dir_type)
        
        # 截取其中一部分
        dataset = torch.utils.data.Subset(dataset, range(args.sample_start_idx, args.sample_end_idx))
        print("len dataset:", len(dataset))
        
        res = evaluate_VQA(model, dataset, args.model_name, target_dataset, time, \
                batch_size=args.batchsize, answer_path=args.answer_path, day_subdir_path=args.day_subdir_path, save_in_progress=True)
        # assert res < 0
        # print(f"res:{res}")
        result[target_dataset] = res
        
    if args.eval_ocrsft or args.eval_all:
        target_dataset = "OCRsft"
        
        dataset = OCRsftDataset(path=args.ocrsft_dataset_path)
        print("len dataset:", len(dataset))
        
        res = evaluate_VQA(model, dataset, args.model_name, target_dataset, time, \
                batch_size=args.batchsize, answer_path=args.answer_path, day_subdir_path=None, save_in_progress=False)
        
    if args.eval_ocrcode or args.eval_all:
        target_dataset = "OCRcode"
        dataset = OCRcodeDataset(path=args.ocrcode_dataset_path)
    
        res = evaluate_VQA(model, dataset, args.model_name, target_dataset, time, \
                batch_size=args.batchsize, answer_path=args.answer_path, day_subdir_path=None, save_in_progress=False)
        
        result[target_dataset] = res
        
    # 0519 new: evaluation for llavabench multilingual
    if args.eval_llavabench_multilingual or args.eval_all:
        target_dataset = "LLaVABenchMultilingual"
        dataset = LlavaBenchMultilingualDataset(ann_dir='/home/lihaoyu/code/0516/llava_bench/imgs',
                                                    gpt_responses_dir = "/home/lihaoyu/code/0516/llava_bench/gpt_responses")
        
        res = evaluate_VQA(model, dataset, args.model_name, target_dataset, time, \
                batch_size=args.batchsize, answer_path=args.answer_path)
        
        result[target_dataset] = res
        
    # 0513 new: 将 codellama 等模型评测 d4j 数据集的代码整合到 vqa_eval 中
    if args.eval_d4j:
        target_dataset = "Defects4J"
        
        # 从指定路径获取需要推理的 buggy function 数据
        dataset = d4jDataset(ann_path=os.path.join(args.data_dir, args.test_file))
        
        # 根据 args 确定输出路径
        if args.do_sample and args.do_beam:
            if args.load_in_4bit:
                output_file_name = f"inference_output_model={args.model_name}_task={args.task_type}_sfttype={args.sft_type}_sftepoch={args.sft_epoch}_dataset={target_dataset}_maxin={args.max_input_len}_maxout={args.max_output_len}_beams={args.num_beams}_4bit.jsonl"
            elif args.load_in_8bit:
                output_file_name = f"inference_output_model={args.model_name}_task={args.task_type}_sfttype={args.sft_type}_sftepoch={args.sft_epoch}_dataset={target_dataset}_maxin={args.max_input_len}_maxout={args.max_output_len}_beams={args.num_beams}_8bit.jsonl"
            else:
                output_file_name = f"inference_output_model={args.model_name}_task={args.task_type}_sfttype={args.sft_type}_sftepoch={args.sft_epoch}_dataset={target_dataset}_maxin={args.max_input_len}_maxout={args.max_output_len}_beams={args.num_beams}.jsonl"
        else:
            raise Exception("Wrong situation, cause now we don't support task other than beam search!")
        output_path = os.path.join(args.output_base_dir, args.method, args.task_type, args.sft_type, output_file_name)
        
        # 注意，这里的 evaluate_VQA 函数针对 beam=10 的情况可能输出结果有区别？
        res = evaluate_APR(model, dataset, args.model_name, target_dataset, \
                batch_size=args.batchsize, output_path=output_path)
    
        result[target_dataset] = res
    
    # 0515 new: 整合 cwe_inf 模块？
    if args.eval_cweinf:
        target_dataset = args.cwe_dataset_name
        
        # 构建 CWE-inference 数据集
        dataset = cweinfDataset(dataset_type=target_dataset, ann_path=os.path.join(args.data_dir, args.test_file))
        
        # 根据 args 确定输出路径
        if args.do_sample and args.do_beam:
            if args.load_in_4bit:
                output_file_name = f"inference_output_model={args.model_name}_task={args.task_type}_sfttype={args.sft_type}_sftepoch={args.sft_epoch}_dataset={target_dataset}_maxin={args.max_input_len}_maxout={args.max_output_len}_beams={args.num_beams}_4bit.jsonl"
            elif args.load_in_8bit:
                output_file_name = f"inference_output_model={args.model_name}_task={args.task_type}_sfttype={args.sft_type}_sftepoch={args.sft_epoch}_dataset={target_dataset}_maxin={args.max_input_len}_maxout={args.max_output_len}_beams={args.num_beams}_8bit.jsonl"
            else:
                output_file_name = f"inference_output_model={args.model_name}_task={args.task_type}_sfttype={args.sft_type}_sftepoch={args.sft_epoch}_dataset={target_dataset}_maxin={args.max_input_len}_maxout={args.max_output_len}_beams={args.num_beams}.jsonl"
        else:
            raise Exception("Wrong situation, cause now we don't support task other than beam search!")
        output_path = os.path.join(args.output_base_dir, args.method, args.task_type, args.sft_type, output_file_name)
    
        # 注意，这里的 evaluate_VQA 函数针对 beam=10 的情况可能输出结果有区别？
        res = evaluate_CWEINF(model, dataset, args.model_name, target_dataset, \
                batch_size=args.batchsize, output_path=output_path)
    
        result[target_dataset] = res
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return None

    # 主进程负责输出最终结果？
    result_path = os.path.join(os.path.join(args.answer_path, args.model_name, args.day_subdir_path), 'result.json')
    
    # 判断一下有没有输出的必要！
    output_flag = False
    for k, v in result.items():
        if v > 0.0:
            output_flag = True
            break
    
    if output_flag:
        with open(result_path, "w") as f:
            f.write(json.dumps(result, indent=4))


if __name__ == "__main__":
    args = parse_args()

    main(args)