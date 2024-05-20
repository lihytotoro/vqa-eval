import json
import os
import random
import re
import sys

import jsonlines
from torch.utils.data import Dataset

sys.path.append("/home/lihaoyu/code/0420")
from instruct import select_instruction

def prompt_processor(prompt):
    if prompt.startswith('OCR tokens: '):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
        if prompt.startswith('Reference OCR token:'):
            question = prompt.split('\n')[1]
        else:
            question = prompt.split('\n')[0]
    elif len(prompt.split('\n')) == 2:
        question = prompt.split('\n')[0]
    else:
        assert False

    return question.lower()
    
class textVQADataset(Dataset):
    def __init__(
        self,
        image_dir_path="./data/TextVQA/train_images",
        ann_path="./data/TextVQA/TextVQA_0.5.1_val.json",
    ):
        self.data = json.load(open(ann_path, "r"))["data"]
        self.image_dir_path = image_dir_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx]['question']
        answers = self.data[idx]['answers']
        img_id = self.data[idx]['image_id']
        qid = self.data[idx]['question_id']
        img_path = os.path.join(self.image_dir_path, f"{img_id}.jpg")
        
        item = {
            "question_id": qid,
            "image_path": img_path,
            "question": question,
            "gt_answers": answers
        }
        
        # if img_id in self.ocr_token_data:
        #     item["ocr_tokens"] = self.ocr_token_data[img_id]['ocr_tokens']
        
        return item
    
class docVQADataset(Dataset):
    def __init__(
        self,
        image_dir_path= "./data/DocVQA/",
        ann_path= "./data/DocVQA/val_v1.0_withQT.json",
        ocr_token_path=None
    ):

        self.data = json.load(open(ann_path, "r"))["data"]
        self.image_dir_path = image_dir_path
        self.ann_path = ann_path
        if ocr_token_path:
            self.ocr_token_data = {item['image_id']: item for item in json.load(open(ocr_token_path, "r"))["data"]}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question_id = self.data[idx]['questionId']  
        relative_img_path = self.data[idx]['image']
        corrected_relative_img_path = relative_img_path.replace("documents", "images")
        img_path = os.path.join(self.image_dir_path, corrected_relative_img_path)
        question = self.data[idx]['question']
        answers = self.data[idx]['answers']
        
        question_type = self.data[idx]['question_types']
        
        return {
            "question_id": question_id,  
            "image_path": img_path,
            "question": question,
            "gt_answers": answers,
            'question_type': question_type,
        }


class docVQATESTDataset(Dataset):
    def __init__(
        self,
        image_dir_path= "./data/DocVQA/",
        ann_path= "./data/DocVQA/val_v1.0_withQT.json",
        ocr_token_path=None
    ):

        self.data = json.load(open(ann_path, "r"))["data"]
        self.image_dir_path = image_dir_path
        self.ann_path = ann_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question_id = self.data[idx]['questionId']  
        relative_img_path = self.data[idx]['image']
        corrected_relative_img_path = relative_img_path.replace("documents", "images")
        img_path = os.path.join(self.image_dir_path, corrected_relative_img_path)
        question = self.data[idx]['question']
        
        
        return {
            "question_id": question_id,  
            "image_path": img_path,
            "question": question,
            "gt_answers": "",
            'question_type': "",
        }
  
class ocrVQADataset(Dataset):
    def __init__(
        self,
        image_dir_path= "./data/ocrvqa/images",
        ann_path= "/data/public/multimodal/multimodal_data/OCR_eval/ocrvqa/ocrvqa_val.jsonl",
    ):
        ann_path= "/data/public/multimodal/multimodal_data/OCR_eval/ocrvqa/ocrvqa_test.jsonl"
        self.image_list = []
        self.question_list = []
        self.answer_list = []

        with jsonlines.open(ann_path) as reader:
            for obj in reader:
                image_ori_path = obj["image"]
                new_base_path = "/data/public/multimodal/multimodal_data/OCR_eval/ocrvqa/images/"
                filename = image_ori_path.split("/")[-1]
                image_file = new_base_path + filename
                question = obj["question"]
                gt_answers = obj["answer"]

                self.image_list.append(image_file)
                self.answer_list.append(gt_answers)
                self.question_list.append(question)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]

        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers
        }
    
class STVQADataset(Dataset):
    def __init__(
        self,
        image_dir_path= "./data/STVQA",
        ann_path= "./data/STVQA/train_task_3.json",
    ):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        data = json.load(open(ann_path, "r"))
        for i in range(len(data['data'])):
            image_path = image_dir_path+'/'+data['data'][i]['dataset']+'/'+data['data'][i]['file_name']
            self.image_list.append(image_path)
            self.answer_list.append(data['data'][i]['answers'])
            self.question_list.append(data['data'][i]['question'])
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers
        }
    
class VQAv2Dataset(Dataset):
    def __init__(
        self,
        image_dir_path="/data/public/multimodal/multimodal_data/coco/coco2017",
        ann_path="./data/VQAv2/v2_mscoco_val2014_annotations.json",
        ques_path="./data/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json",
    ):
        self.annotations = json.load(open(ann_path, "r"))['annotations']
        self.questions = json.load(open(ques_path, "r"))['questions']
        self.image_dir_path = image_dir_path
        self.ocr_token_data = {}

    def __len__(self):
        return len(self.annotations) 

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        annotation_qid = annotation['question_id']
        # Find the matching question based on question_id
        question_data = next((q for q in self.questions if q['question_id'] == annotation_qid), None)
        if question_data is None:
            print(f"No matching question found for question_id: {annotation_qid}")
            return None
        
        question = question_data['question']
        img_id = str(question_data['image_id']).zfill(12)
        qid = question_data['question_id']
        answers = [ans['answer'] for ans in annotation['answers']] 
        img_path = os.path.join(self.image_dir_path, f"{img_id}.jpg")

        item = {
            "question_id": qid,
            "image_path": img_path,
            "question": question,
            "gt_answers": answers
        }
        return item


class ESTVQADataset(Dataset):
    def __init__(
        self,
        image_dir_path= "./data/ESTVQA/images/train",
        ann_path= "./data/ESTVQA/annotations/train.json",
    ):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        with open(ann_path,'r') as f:
            data = json.load(f)
            for i in range(len(data)):
                image_path = os.path.join(image_dir_path, data[i]['image'])
                for j in range(len(data[i]['annotation'])):
                    question = data[i]['annotation'][j]['question']
                    answer = data[i]['annotation'][j]['answer']
                    self.image_list.append(image_path)
                    self.question_list.append(question)
                    self.answer_list.append(answer)
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers
        }
        
class WebSRCDataset(Dataset):
    def __init__(
        self,
        image_dir_path= "/data/public/multimodal/multimodal_data/OCR_train/WEB/WebSRC",
        ann_path= "/data/public/multimodal/multimodal_data/OCR_train/WEB/WebSRC/dev_entries_dedup.jsonl",
    ):
        self.image_list = []
        # self.q_lsit = [
        #     '请仔细描述一下这幅图片',
        #     '我可以用这个网站做什么',
        #     '这个网站是关于什么的',
        #     '这幅图片是什么',
            
        # ]
        self.q_lsit = [
            '请仔细描述一下这幅图片',
            '请问我可以用这个网站做什么?',
            '我可以在这个网站上做什么',
            '这个网站是关于什么的',
            '这幅图片是什么',
            'What can I do on this website?',
            'Please describe this image',
            'Please describe this picture',
            'What is this website about?',
        ]
        self.question_list = []
        self.answer_list = []
        with jsonlines.open(ann_path) as reader:
            for obj in reader:
                # image_file = obj["img_path"]
                # image_file = image_dir_path + image_file
                image_file = obj["img_path"]
                image_file = os.path.join(image_dir_path, image_file)
                question = obj["question"]
                gt_answers = obj["answer"]

                self.image_list.append(image_file)
                self.answer_list.append(gt_answers)
                self.question_list.append(question)
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # question = self.question_list[idx]
        question = random.choice(self.q_lsit)
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers
        }
        
        
class VisualMRCDataset(Dataset):
    def __init__(
        self,
        image_dir_path= "/data/public/multimodal/multimodal_data/OCR_train/WEB/VisualMRC_official",
        ann_path= "/data/public/multimodal/multimodal_data/OCR_train/WEB/VisualMRC_official/qa_val.jsonl",
    ):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        self.q_lsit = [
            '请仔细描述一下这幅图片',
            '请问我可以用这个网站做什么?',
            '我可以在这个网站上做什么',
            '这个网站是关于什么的',
            '这幅图片是什么',
            'What can I do on this website?',
            'Please describe this image',
            'Please describe this picture',
            'What is this website about?',
        ]
        with jsonlines.open(ann_path) as reader:
            for obj in reader:
                image_file = obj["img_path"]
                image_file = os.path.join(image_dir_path, image_file)
                question = obj["question"]
                gt_answers = obj["answer"]

                self.image_list.append(image_file)
                self.answer_list.append(gt_answers)
                self.question_list.append(question)
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # question = self.question_list[idx]
        # question = '请仔细描述一下这幅图片'
        question = random.choice(self.q_lsit)
        # random choice + 'Grounding' or ''
        add = random.choice([' Grounding all object mentioned in your answer.', ''])
        question = question + add
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers
        }
        
class CaseDataset(Dataset):
    def __init__(
        self,
        image_dir_path= "/home/cuijunbo/0328/data/scanbench/image",
        ann_path= "/home/cuijunbo/0328/data/webagent_chat-1700_scanbench.json",
    ):
        with open(ann_path, "r", encoding='utf8') as f:
            self.data = json.loads(f.read())
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        
        for item in self.data:
            image_list = f"{image_dir_path}/{item['id']}.png"
            self.image_list.append(image_list)
            question_list = item["question"]
            self.question_list.append(question_list)
            self.answer_list.append(item["gpt_a"])
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        question = self.question_list[idx]
        # question = random.choice(self.q_lsit)
        # add = random.choice(['\nGrounding all object in this web page.', ''])
        add = ''
        question = question + add
        answers = self.answer_list[idx]
        img_path = self.image_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answers
        }
        
# 0329 new: 用于构造 MathVista 和 MMVet 数据集相关的数据集（应当类似 CaseDataset，包含 image_path、question、gt_answers？其中 gt_answers 不一定需要）
class GroundingDataset(Dataset):
    # target_samples_path: 指定某个数据集中取出哪些样例进行推理测试
    def __init__(
        self,
        bbox_dir="/home/lihaoyu/code/0411/valid_samples_bbox_all_red",
        bbox_dir_type="layers"
    ):
        # 记录每张图片所在路径
        self.image_list = []
        
        # 展开 bbox_dir 下的所有 png 路径
        # v1: version; v2: day; v3: hour;
        if bbox_dir_type == "layers":
            bbox_subdirs_v1 = [os.path.join(bbox_dir, version_subdir) for version_subdir in os.listdir(bbox_dir)]
            assert len(bbox_subdirs_v1) == 2
            bbox_subdirs_v2 = []
            for bbox_subdir_v1 in bbox_subdirs_v1:
                bbox_subdirs_v2.extend([os.path.join(bbox_subdir_v1, day_subdir) for day_subdir in os.listdir(bbox_subdir_v1)])
            assert len(bbox_subdirs_v2) == 5
            bbox_subdirs_v3 = []
            for bbox_subdir_v2 in bbox_subdirs_v2:
                bbox_subdirs_v3.extend([os.path.join(bbox_subdir_v2, hour_subdir) for hour_subdir in os.listdir(bbox_subdir_v2)])
            for bbox_subdir_v3 in bbox_subdirs_v3:
                self.image_list.extend([os.path.join(bbox_subdir_v3, img_path) for img_path in os.listdir(bbox_subdir_v3)])
            # 这里其实最开始的时候确实是 668532 而不是 668511
            assert len(self.image_list) == 668532
        elif bbox_dir_type == "single":
            # 只有单层目录
            self.image_list.extend([os.path.join(bbox_dir, image_file) for image_file in os.listdir(bbox_dir)])
            assert len(self.image_list) == 1729
    
    # 注意：这里应该不是去跑整个数据集，而是针对这里提到的两个数据集，只访问它们的子集
    # 因此，返回的长度应当是在 dataset 类中重新创建的列表，而不是直接使用 img_list 等的长度
    
    def __len__(self):
        return len(self.image_list)
        
    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        # 0403 new：在每个问题后面都加上详细的需求？
        question = "please identify the main content of the web page and give me the bounding box of the main content with the format of <box>left top right bottom</box>"
        # 0403 new：这里我们可以给出每个问题对应的 ground truth 答案
        answer = None
        
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": answer,
        }
        
class OCRsftDataset(Dataset):
    def __init__(
        self,
        path="/home/lihaoyu/code/0420/all_test_data/test_dataset.jsonl"
    ):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        
        subset_to_category_dict = {'web':'web', 'VisualMRC':'web', 'CCpdf':'doc', 'DocVQA':'doc', \
            'KleisterCharity':'doc', 'InfographicsVQA':'infographic', 'ChartQA':'chart', 'TURL':'table', 'TabFact':'table', 'TextVQA':'photograph'}
        
        with jsonlines.open(path, "r") as reader:
            for item in reader:
                self.image_list.append(item["img_path"])
                dataset_name = item['dataset_name']
                instruction = select_instruction(image_context=subset_to_category_dict[dataset_name])
                self.question_list.append(instruction)
                self.answer_list.append(item["gt_answer"])
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        question = self.question_list[idx]
        gt_answer = self.answer_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": gt_answer
        }


class OCRcodeDataset(Dataset):
    def __init__(
        self,
        path="/home/cuijunbo/LMUData/images/MME"
    ):
        self.image_list = []
        self.question_list = []
        self.answer_list = []
        
        all_image_name_list = os.listdir(path)
        image_name_list = []
        for image_name in all_image_name_list:
            try:
                if image_name.endswith(".jpg"):
                    prefix = int(image_name.removesuffix(".jpg"))
                elif image_name.endswith(".png"):
                    prefix = int(image_name.removesuffix(".png"))
                else:
                    continue
                if prefix >= 780 and prefix <= 819:
                    image_name_list.append(image_name)
            except:
                continue
        
        # assert len(image_name_list) == 50
        print(f"len dataset ocrcode:{len(image_name_list)}")
        
        for image_name in image_name_list:
            image_path = os.path.join(path, image_name)
            
            instruction = select_instruction('doc')
            
            self.image_list.append(image_path)
            self.question_list.append(instruction)
            self.answer_list.append("")
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        question = self.question_list[idx]
        gt_answer = self.answer_list[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": gt_answer
        }

# class CasematDataset(Dataset):
#     def __init__(
#         self,
#         image_dir_path= "/data/public/multimodal/cuijunbo/tmp/zhihu/vqa_eval/image",
#         ann_path= "/data/public/multimodal/cuijunbo/tmp/zhihu/vqa_eval/image/sphx_glr_lifecycle_010_2_00x.txt",
#     ):
        
#         self.image_list = []
#         self.question_list = []
        
#     def __len__(self):
#         return len(self.image_list)

#     def __getitem__(self, idx):
#         question = self.question_list[idx]
#         img_path = self.image_list[idx]
#         return {
#             "image_path": img_path,
#             "question": question,
#             "gt_answers": ''
#         }

class CasematDataset(Dataset):
    def __init__(
        self,
        image_dir_path="/data/public/multimodal/cuijunbo/tmp/zhihu/vqa_eval/image",
        ann_path="/data/public/multimodal/cuijunbo/tmp/zhihu/vqa_eval/image/sphx_glr_lifecycle_010_2_00x.txt",
    ):
        self.image_list = []
        self.question_list = []
        
        for img_file in os.listdir(image_dir_path):
            if img_file.endswith(".png"):
                self.image_list.append(os.path.join(image_dir_path, img_file))
        
        with open(ann_path, 'r') as file:
            self.question_list = file.read().splitlines()
        
        self.image_question_pairs = [(img, q) for img in self.image_list for q in self.question_list]
        
    def __len__(self):
        return len(self.image_question_pairs)

    def __getitem__(self, idx):
        img_path, question = self.image_question_pairs[idx]
        return {
            "image_path": img_path,
            "question": question,
            "gt_answers": '' 
        }

# 0426
class chairDataset(Dataset):
    def __init__(
        self,   
        # 后测了这个 ann_path 的结果
        ann_path = '/data/public/multimodal/multimodal_data/dpo/eval/CHAIR/chair_instruction2_img_diversify_8_noemo.jsonl',
        # 先测了这个 ann_path 的结果
        # ann_path = '/data/public/multimodal/multimodal_data/dpo/eval/MMHal/response_template_addpath_qy.jsonl'
    ):
        with jsonlines.open(ann_path) as reader:
            self.data = []
            for obj in reader:
                self.data.append(obj)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        question = self.data[idx]['question']
        image_path = self.data[idx]['image']
        return {
            "image_path": image_path,
            "question": question,
            "gt_answers": '',
        }


# 0519 new: 多语问答数据集（包含 French, German, Portuguese, Spanish）
class LlavaBenchMultilingualDataset(Dataset):
    def __init__(
        self, 
        # 存放所有语言数据集的路径，下面有四个子目录（四种语言）
        ann_dir = '/home/lihaoyu/code/0516/llava_bench/imgs',
        # 使用 gpt4 生成的 gt 所在路径
        gpt_responses_dir = "/home/lihaoyu/code/0516/llava_bench/gpt_responses",
    ):
        # 从 4 * 60 = 240 个样例中随机选取 4 * 6 = 24 个不同的 sample（最好保证图片不重复，但也不必须）
        langs = os.listdir(ann_dir)
        assert len(langs) == 4
        
        self.data = []
        for lang in langs:
            lang_dir = os.path.join(ann_dir, lang)
            # 从中随机选取 6 个样例
            # selected_idxs = list(range(60))
            selected_idxs = random.sample(range(60), 6)
            # selected_idxs.sort()
            for selected_idx in selected_idxs:
                selected_img_path = os.path.join(lang_dir, f"{lang}_{selected_idx}.jpg")
                selected_que_path = os.path.join(lang_dir, f"{lang}_{selected_idx}.txt")
                selected_ans_path = os.path.join(gpt_responses_dir, lang, f"{lang}_{selected_idx}_gpt4_ans.txt")
                with open(selected_que_path, "r", encoding="utf-8") as fq:
                    selected_que = fq.read().strip()
                with open(selected_ans_path, "r", encoding="utf-8") as fa:
                    selected_ans = fa.read().strip()
                self.data.append({"lang":lang, "sample_idx":selected_idx, "image_path":selected_img_path, "question":selected_que, "gt_answer":selected_ans})
            
    
    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, idx):
        # lang = self.data[idx]['lang']
        # sample_idx = self.data[idx]['sample_idx']
        question = self.data[idx]['question']
        image_path = self.data[idx]['image_path']
        gt_answer = self.data[idx]['gt_answer']
        return {
            # "lang": lang, 
            # "sample_idx": sample_idx, 
            "image_path": image_path,
            "question": question,
            "gt_answers": gt_answer
        }