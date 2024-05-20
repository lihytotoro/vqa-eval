import requests
import torch
from PIL import Image
from transformers import (Pix2StructForConditionalGeneration,
                          Pix2StructProcessor)


class Pix2Struct:
    def __init__(self, model_path: str, device: str = "cuda", font_path: str = None):
        self.model = Pix2StructForConditionalGeneration.from_pretrained(model_path).to(device)
        self.processor = Pix2StructProcessor.from_pretrained(model_path)
        #self.processor.tokenizer.padding_side = 'left'
        self.device = device
        self.font_path = font_path
        
    def generate(self, images, questions):
        # Ensure the length of images and questions are the same
        assert len(images) == len(questions), "Mismatched number of images and questions."

        # Convert single image path to Pillow Image for each image in the batch
        pil_images = [Image.open(img_path) for img_path in images]

        # Process all image-question pairs in the batch at once
        if self.font_path is None:
            inputs = self.processor(images=pil_images, text=questions, return_tensors="pt", padding='longest').to(self.device)
        else :
            inputs = self.processor(images=pil_images, text=questions, return_tensors="pt", padding='longest', font_path=self.font_path).to(self.device)


        with torch.inference_mode():
        # Generate predictions for the whole batch
        # 我完全不知道我当时哪里来的这段代码，为什么是这个参数，但是应该是复现了别人的结果的
            predictions = self.model.generate(**inputs,
                                        do_sample=False,
                                        num_beams=1,
                                        max_new_tokens = 100,
                                        min_new_tokens = 1,
                                        top_p=0.9,
                                        repetition_penalty=1.5,
                                        length_penalty=1.0,
                                        temperature=1)
        
        # Decode all the predictions
        answers = [self.processor.decode(pred, skip_special_tokens=True) for pred in predictions]

        return answers