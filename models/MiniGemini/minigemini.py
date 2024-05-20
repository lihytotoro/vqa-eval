import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
import requests
from io import BytesIO
from transformers import TextStreamer
# import argparse

try:
    from diffusers import StableDiffusionXLPipeline
except:
    print('please install diffusers==0.26.3')

try:
    from paddleocr import PaddleOCR
except:
    print('please install paddleocr following https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/README_en.md')

import sys

from mgm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from mgm.conversation import conv_templates, SeparatorStyle
from mgm.model.builder import load_pretrained_model
from mgm.utils import disable_torch_init
from mgm.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

class MiniGemini():
    def __init__(self, model_path, device=None):
        
        self.model_path = model_path
        self.model_base = None
        self.conv_mode = "gemma"
        self.temperature = 0.2
        self.max_new_tokens = 512
        self.load_8bit = False
        self.load_4bit = False
        
        disable_torch_init()
        self.model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(self.model_path, \
            self.model_base, self.model_name, self.load_8bit, self.load_4bit, device=device)
    
    # 用于从指定路径载入图像
    def load_image(self, image_file):
        if image_file.startswith('http://') or image_file.startswith('https://'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image

    # 直接使用 github-minigemini 的 cli.py
    def generate(self, images, questions, datasetname=None):
        
        # if args.ocr and args.image_file is not None:
        #     ocr = PaddleOCR(use_angle_cls=True, use_gpu=True, lang="ch")
        #     result = ocr.ocr(args.image_file)   
        #     str_in_image = ''
        #     if result[0] is not None:
        #         result = [res[1][0] for res in result[0] if res[1][1] > 0.1]
        #         if len(result) > 0:
        #             str_in_image = ', '.join(result)
        #             print('OCR Token: ' + str_in_image)
        
        # 我要评测的是 2b 模型，也就是 gemma
        conv_mode = "gemma"
        
        conv = conv_templates[conv_mode].copy()
        # if "mpt" in model_name.lower():
        #     roles = ('user', 'assistant')
        # else:
        #     roles = conv.roles
        roles = conv.roles

        # 从这里输入图片
        if images is not None:
            
            image_convert = []
            for _image in images:
                image_convert.append(self.load_image(_image))
        
            # 对图片进行处理？
            if hasattr(self.model.config, 'image_size_aux'):
                if not hasattr(self.image_processor, 'image_size_raw'):
                    self.image_processor.image_size_raw = self.image_processor.crop_size.copy()
                self.image_processor.crop_size['height'] = self.model.config.image_size_aux
                self.image_processor.crop_size['width'] = self.model.config.image_size_aux
                self.image_processor.size['shortest_edge'] = self.model.config.image_size_aux
            
            # Similar operation in model_worker.py
            image_tensor = process_images(image_convert, self.image_processor, self.model.config)
        
            image_grid = getattr(self.model.config, 'image_grid', 1)
            if hasattr(self.model.config, 'image_size_aux'):
                raw_shape = [self.image_processor.image_size_raw['height'] * image_grid,
                            self.image_processor.image_size_raw['width'] * image_grid]
                image_tensor_aux = image_tensor 
                image_tensor = torch.nn.functional.interpolate(image_tensor,
                                                            size=raw_shape,
                                                            mode='bilinear',
                                                            align_corners=False)
            else:
                image_tensor_aux = []

            if image_grid >= 2:            
                raw_image = image_tensor.reshape(3, 
                                                image_grid,
                                                self.image_processor.image_size_raw['height'],
                                                image_grid,
                                                self.image_processor.image_size_raw['width'])
                raw_image = raw_image.permute(1, 3, 0, 2, 4)
                raw_image = raw_image.reshape(-1, 3,
                                            self.image_processor.image_size_raw['height'],
                                            self.image_processor.image_size_raw['width'])
                        
                if getattr(self.model.config, 'image_global', False):
                    global_image = image_tensor
                    if len(global_image.shape) == 3:
                        global_image = global_image[None]
                    global_image = torch.nn.functional.interpolate(global_image, 
                                                                size=[self.image_processor.image_size_raw['height'],
                                                                        self.image_processor.image_size_raw['width']], 
                                                                mode='bilinear', 
                                                                align_corners=False)
                    # [image_crops, image_global]
                    raw_image = torch.cat([raw_image, global_image], dim=0)
                image_tensor = raw_image.contiguous()
                image_tensor = image_tensor.unsqueeze(0)
        
            if type(image_tensor) is list:
                image_tensor = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor]
                image_tensor_aux = [image.to(self.model.device, dtype=torch.float16) for image in image_tensor_aux]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
                image_tensor_aux = image_tensor_aux.to(self.model.device, dtype=torch.float16)

        # 由于只进行一轮对话，因此 while 可以暂时去掉

        inp = questions[0]
        inp = f"{inp}\nAnswer the question using a single word or phrase."

        # if args.ocr and len(str_in_image) > 0:
        #     inp = inp + '\nReference OCR Token: ' + str_in_image + '\n'

        if images is not None:
            # first message
            if self.model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = (DEFAULT_IMAGE_TOKEN + '\n')*len(images) + inp
            conv.append_message(conv.roles[0], inp)
            # images = None

        # print(inp)

        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # print(prompt)
        
        # add image split string
        if prompt.count(DEFAULT_IMAGE_TOKEN) >= 2:
            final_str = ''
            sent_split = prompt.split(DEFAULT_IMAGE_TOKEN)
            for _idx, _sub_sent in enumerate(sent_split):
                if _idx == len(sent_split) - 1:
                    final_str = final_str + _sub_sent
                else:
                    final_str = final_str + _sub_sent + f'Image {_idx+1}:' + DEFAULT_IMAGE_TOKEN
            prompt = final_str
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)
        # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            # 生成回复！
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                images_aux=image_tensor_aux if len(image_tensor_aux)>0 else None,
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                bos_token_id=self.tokenizer.bos_token_id,  # Begin of sequence token
                eos_token_id=self.tokenizer.eos_token_id,  # End of sequence token
                pad_token_id=self.tokenizer.pad_token_id,  # Pad token
                # streamer=streamer,
                use_cache=True)

        # 此即最终回复，应该返回，并且外面应该包一层列表
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        print(outputs)
        
        return [outputs]
