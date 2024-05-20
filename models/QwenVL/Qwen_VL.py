import os
import sys

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

ds_collections = {
    'textVQA': {
        'max_new_tokens': 10,
    },
    'ocrVQA': {
        'max_new_tokens': 100,
    },
    'STVQA': {
        'max_new_tokens': 10,
    },
    'VQAv2': {
        'max_new_tokens': 10,
    },
    'keepVQA':{
        'max_new_tokens': 100,

    },
    'FUNSD':{
        'max_new_tokens' : 1000,
    },
    'YFVQA':{
        'max_new_tokens' : 10,
    },
    'ocr':{
        'max_new_tokens' : 1000,
    },
    'vqav2_val': {
        'train': 'data/vqav2/vqav2_train.jsonl',
        'test': 'data/vqav2/vqav2_val.jsonl',
        'question': 'data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'data/vqav2/v2_mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vqav2_testdev': {
        'train': 'data/vqav2/vqav2_train.jsonl',
        'test': 'data/vqav2/vqav2_testdev.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'okvqa_val': {
        'train': 'data/okvqa/okvqa_train.jsonl',
        'test': 'data/okvqa/okvqa_val.jsonl',
        'question': 'data/okvqa/OpenEnded_mscoco_val2014_questions.json',
        'annotation': 'data/okvqa/mscoco_val2014_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'textvqa_val': {
        'train': 'data/textvqa/textvqa_train.jsonl',
        'test': 'data/textvqa/textvqa_val.jsonl',
        'question': 'data/textvqa/textvqa_val_questions.json',
        'annotation': 'data/textvqa/textvqa_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_val': {
        'train': 'data/vizwiz/vizwiz_train.jsonl',
        'test': 'data/vizwiz/vizwiz_val.jsonl',
        'question': 'data/vizwiz/vizwiz_val_questions.json',
        'annotation': 'data/vizwiz/vizwiz_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
    'vizwiz_test': {
        'train': 'data/vizwiz/vizwiz_train.jsonl',
        'test': 'data/vizwiz/vizwiz_test.jsonl',
        'metric': None,
        'max_new_tokens': 10,
    },
    'docVQA': {
        'train': 'data/docvqa/train.jsonl',
        'test': 'data/docvqa/val.jsonl',
        'annotation': 'data/docvqa/val/val_v1.0.json',
        'metric': 'anls',
        'max_new_tokens': 100,
    },
    'docvqa_test': {
        'train': 'data/docvqa/train.jsonl',
        'test': 'data/docvqa/test.jsonl',
        'metric': None,
        'max_new_tokens': 100,
    },
    'chartqa_test_human': {
        'train': 'data/chartqa/train_human.jsonl',
        'test': 'data/chartqa/test_human.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'chartqa_test_augmented': {
        'train': 'data/chartqa/train_augmented.jsonl',
        'test': 'data/chartqa/test_augmented.jsonl',
        'metric': 'relaxed_accuracy',
        'max_new_tokens': 100,
    },
    'gqa_testdev': {
        'train': 'data/gqa/train.jsonl',
        'test': 'data/gqa/testdev_balanced.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    },
    'ocrvqa_val': {
        'train': 'data/ocrvqa/ocrvqa_train.jsonl',
        'test': 'data/ocrvqa/ocrvqa_val.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ocrvqa_test': {
        'train': 'data/ocrvqa/ocrvqa_train.jsonl',
        'test': 'data/ocrvqa/ocrvqa_test.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 100,
    },
    'ai2diagram_test': {
        'train': 'data/ai2diagram/train.jsonl',
        'test': 'data/ai2diagram/test.jsonl',
        'metric': 'accuracy',
        'max_new_tokens': 10,
    }
}

class QwenVL:
    def __init__(self, model_path, device) -> None:
        sys.path.append(model_path) 
        
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)
        # print(f"model_path: {self.model.generation_config}")
        
        # self.model.generation_config.top_p = 0.01
        # print(f"model_path: {self.model.generation_config}")
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token_id = self.tokenizer.eod_id
        print(f"device: {device}")
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    # def generate(self, datasetname,image, question):
        
    #     # query = self.tokenizer.from_list_format([
    #     #     {'image': image[0]},
    #     #     {'text': question[0]+' answer:'},
    #     # ])
    #     query = f'<img>{image[0]}</img>\n{question[0]}'
    #     response, _ = self.model.chat(self.tokenizer, query=query, history=None)
    #     return [response]
    
    def generate(self, datasetname, images, questions):
        # image = image[0]
        # question = question[0]
        try:
            max_new_tokens=ds_collections[datasetname]['max_new_tokens']
        except:
            max_new_tokens=2048

        prompt = '<img>{}</img>{} Answer:'
        # prompt = '<img>{}</img>{} answer:'
        # prompt = '<img>{}</img>OCR with grounding: '

        # input_texts = [prompt.format(img, q) for img, q in zip(image, question)]
        input_texts = [prompt.format(img, q) for img, q in zip(images, questions)]
        print(f"input_texts: {input_texts}")
        # Tokenize
        # inputs = self.tokenizer.encode_plus(
        #     input_text, 
        #     return_tensors="pt", 
        #     add_special_tokens=True
        # )
        inputs = self.tokenizer(input_texts, return_tensors='pt', padding='longest')
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Generate
        pred = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens, # Example, adjust accordingly
            min_new_tokens=1,
            length_penalty=1,
            num_return_sequences=1,
            output_hidden_states=True,
            use_cache=True,
            pad_token_id=self.tokenizer.eod_id,
            eos_token_id=self.tokenizer.eod_id,
        )
        
        # answer = self.tokenizer.decode(pred[0], skip_special_tokens=True).strip()
        # index = answer.index('Answer:') + 8  # 加8是为了跳过"Answer: "字符串
        # answer = answer[index:]
        answer = [self. tokenizer.decode(_[input_ids.size(1):].cpu(),
                    skip_special_tokens=True).strip() for _ in pred]
        # answer = ["<ref>"+self. tokenizer.decode(_[input_ids.size(1):].cpu(),
                    # skip_special_tokens=False).strip() for _ in pred]
        # answer = self. tokenizer.decode(pred[input_ids.size(1):].cpu(),
                    # skip_special_tokens=False).strip()
        print(f"answer: {answer}")
        # answer = answer[0]
                    
        return answer

'''
/home/cuijunbo/miniconda3/envs/minicpm-v/bin/torchrun --nproc_per_node=${NPROC_PER_NODE:-8} \
        --nnodes=${WORLD_SIZE:-1} \
        --node_rank=${RANK:-0} \
        --master_addr=${MASTER_ADDR:-127.0.0.1} \
        --master_port=${MASTER_PORT:-12345} \
        eval.py --model_name minicpm --eval_textVQA --batchsize 1 

'''