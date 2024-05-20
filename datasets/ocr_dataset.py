import json
import os
import random
import re
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset


def remove_special_chars(s):
    pattern = r"[^a-zA-Z0-9\s]"
    s = re.sub(pattern, "", s)
    return s

def ocr_instruction_templates():
    # instructions = [
    #     "Identify the text in the image with position.",
    #     "Pinpoint and indicate the text and its location within the image.",
    #     "Find the text in the image and identify its positional.",
    #     "Detect the text within the image and specify its position.",
    #     "Locate the text in the image and detail its position."
    # ]
    web_structured_instructions = [
        "Transcribe the webpage content, following its hierarchical layout, and specify the location of each text section.",
        "Detail the webpage's content, indicating the precise location of text within each structured area.",
        "Extract and categorize the text from the webpage, clearly noting the position of each structural element.",
        "Systematically capture text from the webpage, aligning with its layout and detailing the position of each organizational element.",
        "Organize and record the webpage's text, respecting the content's structure and noting the specific location of each section."
    ]
    new_question = random.choice(web_structured_instructions)

    return new_question

class ocrDataset(Dataset):
    def __init__(
        self,
        image_dir_path= "./data/ocr",
        dataset_name = "ct80"
    ):
        self.image_dir_path = image_dir_path
        self.dataset_name = dataset_name
        file_path = os.path.join(image_dir_path, f'{dataset_name}/test_label.txt')
        file = open(file_path, "r")
        self.lines = file.readlines()
    def __len__(self):
        return len(self.lines)
    def __getitem__(self, idx):
        image_id = self.lines[idx].split()[0]
        img_path = os.path.join(self.image_dir_path,f'{self.dataset_name}/{image_id}')
        answers = self.lines[idx].split()[1]
        return {
            "image_path": img_path,
            "gt_answers": answers}
        
class C4WEBDataset(Dataset):
    def __init__(
        self,
        image_dir_path= "/home/cuijunbo/new/Multimodal/vqa_eval/data/new_c4web/672",
        ann_path = '/home/cuijunbo/new/Multimodal/vqa_eval/data/new_c4web/processed_test.json'
    ):
        self.image_dir_path = image_dir_path
        self.ann_path = ann_path
        self.data = json.load(open(ann_path))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_id = item.get('img_id', "")
        img_path = item.get('img_path', "")
        img_path = img_path.replace('IDLOCR', 'new_c4web')
        answers = item.get('text', "")
        
        return {
            "image_path": img_path,
            "gt_answers": answers,
            "question": ocr_instruction_templates(),
            "question_id": image_id
        }
        
class IDLOCRDataset(Dataset):
    def __init__(
        self,
        image_dir_path= "/home/cuijunbo/new/Multimodal/vqa_eval/data/IDLOCR",
        ann_path = "/home/cuijunbo/new/Multimodal/vqa_eval/data/IDLOCR/test.json",
    ):
        self.image_dir_path = image_dir_path
        self.ann_path = ann_path
        self.data = json.load(open(ann_path))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_id = item.get('img_id', "")
        img_path = item.get('img_path', "")
        answers = item.get('text', "")
        
        return {
            "image_path": img_path,
            "gt_answers": answers,
            "question": ocr_instruction_templates(),
            "question_id": image_id
        }
        
class streetViewOCRDataset(Dataset):
    def __init__(
        self,
        image_dir_path= "/home/cuijunbo/new/Multimodal/vqa_eval/data/streetViewOCR",
        ann_path = "/home/cuijunbo/new/Multimodal/vqa_eval/data/streetViewOCR/test.json",
    ):
        self.image_dir_path = image_dir_path
        self.ann_path = ann_path
        self.data = json.load(open(ann_path))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_id = item.get('img_id', "")
        img_path = item.get('img_path', "")
        answers = item.get('text', "")
        
        return {
            "image_path": img_path,
            "gt_answers": answers,
            "question": ocr_instruction_templates(),
            "question_id": image_id
        }