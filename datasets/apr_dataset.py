import json
import os
import random
import re
import sys

import jsonlines
from torch.utils.data import Dataset


# 将我测试 APR 的数据合并到 vqa_eval 里面来？
class d4jDataset(Dataset):
    '''
    bug_id: proj + id
    buggy: buggy func
    fix: fixed func
    buggy_white_space: 用于 backfill 阶段对齐其他部分的代码
    fixed_white_space
    func_start_idx
    func_end_idx
    prefix: before the buggy hunk
    suffix: after the buggy hunk
    buggy_chunk
    fixed_chunk
    input: input to the model, which contains <FILL_ME> token
    output: gt_answer at buggy_chunk
    '''
    def __init__(
        self,
        ann_path="/data/public/multimodal/lihaoyu/szx/datasets/d4j-processed/processed/defects4j_all_single_func_repairllama_wo_initial_prompt.jsonl",
    ):
        self.data = []
        with jsonlines.open(ann_path) as reader:
            for obj in reader:
                self.data.append(obj)
            
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        bug_id = self.data[idx]['bug_id']
        buggy_input = self.data[idx]['input']
        gt_output = self.data[idx]['output']
        return {
            "bug_id": bug_id, 
            "input": buggy_input,
            "gt_output": gt_output
        }
        

# 这个数据集负责 cwe-inference 任务
class cweinfDataset(Dataset):
    '''
    dataset_type: defects4j or java-juliet，数据集不同，有无 CWE 以及 有无 bug_id 是有区别的
    '''
    def __init__(
        self,
        dataset_type="java-juliet",
        ann_path="/data/public/multimodal/lihaoyu/szx/datasets/d4j-processed/processed/defects4j_all_single_func_repairllama_wo_initial_prompt.jsonl",
    ):
        self.dataset_type = dataset_type
        self.data = []
        with jsonlines.open(ann_path) as reader:
            for obj in reader:
                self.data.append(obj)
    
    # 返回数据集的长度
    def __len__(self):
        return len(self.data)
    
    # 注意这里，不同的数据集有不同的数据处理细节
    # java-juliet-test: /data/public/multimodal/lihaoyu/szx/datasets/java-juliet/src/parsed_dataset/jsonl/cwe-inference/finetuning_data_maxlen=1024_modeltype=qwen_usertype=qlora_dataset=cwe-inference_trainratio=0.9_split=test.jsonl
    # defects4j（无 gt cwe）: 
    def __getitem__(self, idx):
        if self.dataset_type == "Java-Juliet":
            # 获取 bug_id
            bug_id = self.data[idx]["conversation_id"]
            # 获取 input，注意这里还有 system？
            system_prompt = self.data[idx]["system"]
            buggy_input = self.data[idx]["conversation"][0]["human"]
            # 提取 CWE
            gt_response = self.data[idx]["conversation"][0]["assistant"]
            cwe_prefix = "The CWE type of the code is: "
            gt_cwe = gt_response.removeprefix(cwe_prefix).split("--")[0]
            assert gt_cwe.startswith("CWE")
            
            return {
                "bug_id": bug_id, 
                "system": system_prompt, 
                "input": buggy_input, 
                "gt_cwe": gt_cwe
            }
        # 对于 d4j 数据集来讲，做 cwe 推理的时候拿到的只有代码，还需要按照 java-j 数据集的模板进行套用
        elif self.dataset_type == "Defects4J":
            bug_id = self.data[idx]['bug_id']
            
            system_prompt = "You are a powerful automatic program repair assistant with plenty of knowledge about common weakness enumeration(CWE). Provide your answer in Markdown."
            
            # 0515 new: 
            user_input_prefix = "The following java code contains a flaw. Read the code snippet carefully and answer the quetion below.\n"
            user_buggy_code_prefix = "Buggy Code:\n"
            user_question_prefix = "Question:\n"
            user_question_content = "What is the exact CWE(common weakness enumeration) type of the flaw? What is the definition of this CWE type? Give your answer in the same format as the following example.\n"
            user_example_prefix = "Answer Example:\n"
            user_example_content_1 = "The CWE type of the code is: CWE129--Improper Validation of Array Index. "
            user_example_content_2 = "The description of CWE129 is: The product uses untrusted input when calculating or using an array index, but the product does not validate or incorrectly validates the index to ensure the index references a valid position within the array.\n"
            user_your_answer_prefix = "Your Answer:"
            ori_buggy_input = self.data[idx]['input']
            
            buggy_input = user_input_prefix + user_buggy_code_prefix + ori_buggy_input + "\n" + user_question_prefix + user_question_content + user_example_prefix + \
                user_example_content_1 + user_example_content_2 + user_your_answer_prefix
            
            gt_cwe = ""
            return {
                "bug_id": bug_id, 
                "system": system_prompt, 
                "input": buggy_input,
                "gt_cwe": gt_cwe
            }
        else:
            raise Exception(f"Wrong dataset_type {self.dataset_type}!")