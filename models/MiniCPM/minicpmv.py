
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

Image.MAX_IMAGE_PIXELS = 1000000000

max_token  = {
    'docVQA': 100,
    'ocrVQA': 100,
    'textVQA': 100,
    'C4WEB': 2048,
    'IDLOCR': 2048,
    'textOCR': 512,
    "docVQATest": 100,
    'Grounding': 16,
}

class MiniCPM_V:

    def __init__(self, model_path, device = None)->None:
        # self.model_path = '/home/zhanghaoye/MiniCPMV_checkpoints_zhy/DPO_exp/minicpmv_slice_v2_sr4500img/minicpmv_DPO-minicpmv_slice_v2_sr4500img_1200step_bs2_gradacc1_beta0.3-minicpmv_slice_v2_sr4500img-1/checkpoints'
        self.model_path = model_path
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).eval()
        # 这里改成 grounding 模型
        # self.ckpt = "/home/cuijunbo/export/1node_0423-ALL/1node_0423-ALL_epoch_2_ckpt_850/1node_0423-ALL_1641_850.pt"
        # self.state_dict = torch.load(self.ckpt, map_location=torch.device('cpu'))
        # self.model.load_state_dict(self.state_dict)
        
        # self.model = self.model.to(dtype=torch.float32)
        # self.model = self.model.to(dtype=torch.bfloat16)
        self.model = self.model.to(dtype=torch.float16)
        
        self.model.to(device)
        
        # self.model.eval().to(device)
        
        # 查看模型参数，期望是 fp16
        # for param in self.model.parameters():
        #     print(param.dtype)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        torch.cuda.empty_cache()

    def generate(self, images, questions, datasetname):
        image = Image.open(images[0]).convert('RGB')
        # datasetname = datasetname.lower()
        try:
            max_new_tokens = max_token[datasetname]
        except:
            max_new_tokens = 1024
        if (datasetname == 'docVQA') or (datasetname == "docVQATest") :
            prompt = "Answer the question directly with single word." + "\n" + questions[0]
            # # image resize w * 3 h * 3
            # ori_width, ori_height = image.size
            # image = image.resize((image.width /2, image.height /2))
            # print(f"from {ori_width}x{ori_height} to {image.width}x{image.height}")
        elif (datasetname == 'textVQA') :
            prompt = "Answer the question directly with single word." + '\n'+ questions[0]
        elif (datasetname == 'ocrVQA') :
            prompt = "Answer the question directly with single word." + "\n" + questions[0]
            # image resize w * 3 h * 3
            # image = image.resize((image.width * 3, image.height * 3))
            # width, height = image.size
            # if width * height < 1344*1344:
            #     r = width / height
            #     height = int(1344 / math.sqrt(r))
            #     width = int(height * r)
            #     image = image.resize((width, height), Image.Resampling.BICUBIC)
        else:
            prompt = questions[0]
        msgs = [{'role': 'user', 'content': prompt}]
        default_kwargs = dict(
            max_new_tokens=1024,
            sampling=False,
            num_beams=3
        )
        # res, _, _ = self.model.chat(
        res = self.model.chat(
            image=image,
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            **default_kwargs
        )
        
        return [res]
    
    def generate_with_interleaved(self, images, questions, datasetname):
        try:
            max_new_tokens = max_token[datasetname]
        except:
            raise Exception("")
            max_new_tokens = 1024
        
        if (datasetname == 'docVQA') or (datasetname == "docVQATest") :
            prompt = "Answer the question directly with single word."
        elif (datasetname == 'textVQA') :
            prompt = "Answer the question directly with single word."
        elif (datasetname == 'ocrVQA') :
            prompt = "Answer the question directly with single word."
        else:
            prompt = questions[0]
        
        default_kwargs = dict(
            max_new_tokens=max_new_tokens,
            sampling=False,
            num_beams=3
        )
        
        content = []
        message = [
            {'type': 'text', 'value': 'Answer the question directly with single word.'},
            {'type': 'image', 'value': images[0]},
            {'type': 'text', 'value': questions[0]}
        ]
        for x in message:
            if x['type'] == 'text':
                content.append(x['value'])
            elif x['type'] == 'image':
                image = Image.open(x['value']).convert('RGB')
                content.append(image)
        msgs = [{'role': 'user', 'content': content}]

        res = self.model.chat(
            msgs=msgs,
            context=None,
            tokenizer=self.tokenizer,
            **default_kwargs
        )

        if isinstance(res, tuple) and len(res) > 0:
            res = res[0]
        print(f"Q: {content}, \nA: {res}")
        return [res]
    
    # 在调用 grounding 模型时，直接只给一张图片，问题都是固定的
    # 由于是 batch 输入的方式，因此改回上面的 generate 格式
    # dataset_name = Grounding
    # def generate(self, images, questions, datasetname):
    #     img_list = []
    #     msg_list = []
        
    #     if datasetname == "Grounding":
    #         max_new_tokens = 16
    #     else:
    #         max_new_tokens = 1024
        
    #     # if len(images) != 1:
    #     #     print(images)
        
        
    #     for idx in range(len(images)):
    #         img = images[idx]
    #         inp = questions[idx]
    #         try:
    #             image = Image.open(img).convert('RGB')
    #         except:
    #             # 写入读取失败的图片所在目录
    #             with open('/data/public/multimodal/lihaoyu/vqa_eval/answers-0423/minicpm/failed_image_paths.txt', 'a') as f:
    #                 f.write(f"{img}\n")
    #             continue
    #         # datasetname = datasetname.lower()
    #         if (datasetname == 'docVQA') or (datasetname == "docVQATest") :
    #             prompt = "Answer the question directly with single word" + "\n" + inp
    #         elif (datasetname == 'textVQA') :
    #             prompt = "Answer the question directly with single word" + '\n'+ inp
    #         elif (datasetname == 'ocrVQA') :
    #             prompt = "Answer the question directly with single word" + "\n" + inp
    #         else:
    #             # 进行单图 grounding 任务时，所有问题都是一样的
    #             prompt = inp
    #         img_list.append(image)
    #         msg_list.append([{'role': 'user', 'content': prompt}])
    #     # 出现了 bug？为什么这里 max_new_tokens 会访问失败？
    #     # 在 batch_size = 1 的情况下，读取失败的图片会直接 continue，因此 img_list 和 msg_list 都没有 append 操作
    #     default_kwargs = dict(
    #         max_new_tokens=max_new_tokens,
    #         # sampling=False,
    #         num_beams=3
    #     )
    #     # 如果出现了上述的异常情况，那么本次 img_list 和 msg_list 应该是 0？
    #     # print(f"img_list:{img_list}")
    #     # print(f"msg_list:{msg_list}")
    #     # 在异常情况下，直接返回一个空列表？
    #     if len(img_list) == 0 or len(msg_list) == 0:
    #         return []
        
    #     # res, _, _ = self.model.chat(
    #     #     image_list=img_list,
    #     #     msgs_list=msg_list,
    #     #     context=None,
    #     #     tokenizer=self.tokenizer,
    #     #     sampling=False,
    #     #     **default_kwargs
    #     # )
    #     default_kwargs = dict(
    #         max_new_tokens=1024,
    #         sampling=False,
    #         num_beams=3
    #     )
    #     res, _, _ = self.model.chat(
    #         image=image,
    #         msgs=msg_list,
    #         context=None,
    #         tokenizer=self.tokenizer,
    #         **default_kwargs
    #     )
        
    #     # print(f"res:{res}")
        
    #     # print(len(res))
    #     # print(res)
        
    #     return res
'''
python -m torch.distributed.launch \
    --nproc_per_node=${NPROC_PER_NODE:-8} \
    --nnodes=${WORLD_SIZE:-1} \
    --node_rank=${RANK:-0} \
    --master_addr=${MASTER_ADDR:-127.0.0.1} \
    --master_port=${MASTER_PORT:-12345} \
    eval.py --model_name minicpm --eval_docVQA --batchsize 1
    
    eval_VisualMRC
python eval.py --model_name minicpm --eval_docVQA --batchsize 1
'''


'''
/home/cuijunbo/miniconda3/envs/minicpm-v/bin/torchrun --nproc_per_node=${NPROC_PER_NODE:-8} \
        --nnodes=${WORLD_SIZE:-1} \
        --node_rank=${RANK:-0} \
        --master_addr=${MASTER_ADDR:-127.0.0.1} \
        --master_port=${MASTER_PORT:-12345} \
        eval.py --model_name minicpm --eval_docVQA --batchsize 1 
'''