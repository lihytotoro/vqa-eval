import torch
import sys

# #! modify muffin path
# muffin_path = '/home/hongyixin/vqa_eval/models/YiVL'
# sys.path.append(muffin_path)

import os.path as osp
import warnings
from PIL import Image
from models.YiVL.evalkit_smp import get_cache_path, load, dump, splitlen
from huggingface_hub import snapshot_download

"""
You can perform inference of Yi-VL through the following steps:
1. clone the repo https://github.com/01-ai/Yi to path-to-Yi
2. set up the environment and install the required packages in path-to-Yi/VL/requirements.txt
3. set Yi_ROOT in vlmeval/config.py
    Yi_ROOT = path-to-Yi

You are all set now! To run a demo for Yi-VL:
```python
from vlmeval import *
model = supported_VLM['Yi_VL_6B']()
model.generate('apple.jpg', 'What is in this image?')
```
To run evaluation for Yi-VL, use `python run.py --model Yi_VL_6B --data {dataset_list}`
"""


def edit_config(repo_id):
    if not osp.exists(repo_id):
        root = get_cache_path(repo_id)
    else:
        root = repo_id
    assert root is not None and osp.exists(root)
    cfg = osp.join(root, 'config.json')
    data = load(cfg)
    mm_vision_tower = data['mm_vision_tower']
    if mm_vision_tower.startswith('./vit/'):
        data['mm_vision_tower'] = osp.join(root, mm_vision_tower)
        assert osp.exists(data['mm_vision_tower'])
        dump(data, cfg)


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)


# Assuming other necessary imports and utility functions are defined elsewhere as in your initial code snippet

class YiVL:

    INSTALL_REQ = True

    def __init__(self, model_path=None, root=None, **kwargs):
        #! modified model path
        #! 
        # model_path = '/home/hongyixin/VLMEvalKit/data/Yi-VL-6B'
        model_path = model_path
        # model_path = '/home/hongyixin/vqa_eval/models/YiVL/models/Yi-VL-34B'
        if root is None:
            warnings.warn(
                'Please set root to the directory of Yi, '
                'which is cloned from here: https://github.com/01-ai/Yi.'
            )

        self.root = osp.join(root, 'VL')
        sys.path.append(self.root)

        if splitlen(model_path, '/') == 2 and not osp.exists(model_path):
            if get_cache_path(model_path) is None:
                snapshot_download(repo_id=model_path)
            edit_config(model_path)
        elif osp.exists(model_path):
            edit_config(model_path)

        from llava.mm_utils import get_model_name_from_path, load_pretrained_model
        from llava.model.constants import key_info

        disable_torch_init()
        key_info['model_path'] = model_path
        get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path,
            device_map='cpu')
        self.model = self.model.cuda()
        self.conv_mode = 'mm_default'

        kwargs_default = dict(temperature=0.2,
                              num_beams=1,
                              do_sample=False,
                              max_new_tokens=1024,
                              top_p=None)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f'Following kwargs received: {self.kwargs}, will use as generation config. ')

    # def generate(self, image_path, prompt, dataset=None):
    def generate(self, datasetname, images, questions):

        from llava.conversation import conv_templates
        from llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.mm_utils import KeywordsStoppingCriteria, expand2square, tokenizer_image_token

        # qs = DEFAULT_IMAGE_TOKEN + '\n' + questions
        #! modify questions!! 
        #! questions[0] === the original "prompt"
        # qs = DEFAULT_IMAGE_TOKEN + '\n' + 'Answer the question directly with single word' + '\n' + questions[0]
        qs = DEFAULT_IMAGE_TOKEN + '\n' + questions[0]
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        questions[0] = conv.get_prompt()
        
        #! show prompt
        print(questions)
        
        input_ids = (
            tokenizer_image_token(questions[0], self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            .unsqueeze(0)
            .cuda()
        )
        # print(f"images:::::::::::::{images}")
        image = Image.open(images[0])
        if getattr(self.model.config, 'image_aspect_ratio', None) == 'pad':
            if image.mode == 'L':
                background_color = int(sum([int(x * 255) for x in self.image_processor.image_mean]) / 3)
            else:
                background_color = tuple(int(x * 255) for x in self.image_processor.image_mean)
            image = expand2square(image, background_color)
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')[
            'pixel_values'
        ][0]

        stop_str = conv.sep
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        self.model = self.model.to(dtype=torch.bfloat16)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda(),
                stopping_criteria=[stopping_criteria],
                use_cache=True,
                **self.kwargs)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(
                f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids'
            )
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True
        )[0]
        outputs = outputs.strip()

        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        print(outputs)
        return [outputs]

"""
python -m torch.distributed.launch \
    --nproc_per_node=${NPROC_PER_NODE:-8} \
    --nnodes=${WORLD_SIZE:-1} \
    --node_rank=${RANK:-0} \
    --master_addr=${MASTER_ADDR:-127.0.0.1} \
    --master_port=${MASTER_PORT:-12345} \
    eval.py --model_name yivl --eval_VisualMRC --batchsize 1  """
# python -m torch.distributed.launch --nproc_per_node=${NPROC_PER_NODE:-8} --nnodes=${WORLD_SIZE:-1} --node_rank=${RANK:-0} --master_addr=${MASTER_ADDR:-127.0.0.1} --master_port=${MASTER_PORT:-12345} eval.py --model_name yivl --eval_textVQA --batchsize 1