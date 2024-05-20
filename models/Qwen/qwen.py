# Qwen 模型目前主要用来获取 buggy code 对应的 CWE
# 0513 new
# 尝试仿照 minicpm_v 等模型，写出 Qwen 模型的输入输出函数
import os
import sys
import torch
from transformers import AutoModel, AutoTokenizer, GenerationConfig
import copy

sys.path.append("/home/lihaoyu/szx/proj/github-proj/Firefly")
from component.utils import ModelUtils
from component.template import template_dict


# 建立 prompt？
def build_prompt(tokenizer, template, query, history, system=None):
    '''
    tokenizer
    template
    query
    history: 对话历史？可能涉及多轮对话的处理
    '''
    # qwen
    template_name = template.template_name
    system_format = template.system_format
    user_format = template.user_format
    assistant_format = template.assistant_format
    # 注意，使用 qwen 回答 CWE 时，自带 system，需要加入这里
    system = system if system is not None else template.system

    if template_name == 'chatglm2':
        prompt = tokenizer.build_prompt(query, history)
        input_ids = tokenizer.encode(prompt)
    elif template_name == 'chatglm3':
        input_ids = build_prompt_chatglm3(tokenizer, query, history, system)
    else:
        history.append({"role": 'user', 'message': query})
        input_ids = []

        # setting system information
        if system_format is not None:
            # system信息不为空
            if system is not None:
                system_text = system_format.format(content=system)
                input_ids = tokenizer.encode(system_text, add_special_tokens=False)
        # concat conversation
        for item in history:
            role, message = item['role'], item['message']
            if role == 'user':
                message = user_format.format(content=message, stop_token=tokenizer.eos_token)
            else:
                message = assistant_format.format(content=message, stop_token=tokenizer.eos_token)
            tokens = tokenizer.encode(message, add_special_tokens=False)
            input_ids += tokens
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    return input_ids


# 载入 tokenizer
def load_tokenizer(model_name_or_path):
    # config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    # 加载tokenzier
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        use_fast=False
        # llama不支持fast
        # use_fast=False if config.model_type == 'llama' else True
    )

    # 这里，现在我们使用的是 Qwen 模型，会进到这里吗？
    if tokenizer.__class__.__name__ == 'QWenTokenizer':
        tokenizer.pad_token_id = tokenizer.eod_id
        tokenizer.bos_token_id = tokenizer.eod_id
        tokenizer.eos_token_id = tokenizer.eod_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # assert tokenizer.pad_token_id is not None, "pad_token_id should not be None"
    return tokenizer


class Qwen:
    def __init__(self, args, device=None) -> None:
        # 传入模型路径
        self.args = args
        self.model_name = self.args.model_name
        self.model_base_dir = self.args.model_base_dir
        self.template_name = self.args.template_name
        # 默认没有 adapter，全部使用 merged model
        self.adapter_dir = None
        
        # old-method, cwe-inference, qlora/lora/full, 1
        self.method = self.args.method
        self.task_type = self.args.task_type
        self.sft_type = self.args.sft_type
        self.sft_epoch = self.args.sft_epoch
        
        # 默认全部 sft 都训练 1 个 epoch
        if self.args.sft_type in ['lora', 'qlora']:
            model_subdir_name = f"firefly-model={self.model_name}_task={self.task_type}_trainingtype={self.sft_type}_merged"
            self.model_path = os.path.join(self.model_base_dir, self.method, self.task_type, self.sft_type, "merged", model_subdir_name)
        elif self.args.sft_type == "full":
            model_subdir_name = f'firefly-model={self.model_name}_task={self.task_type}_trainingtype={self.sft_type}'
            self.model_path = os.path.join(self.model_base_dir, self.method, self.task_type, self.sft_type, "original", model_subdir_name)
        else:
            raise Exception(f"Wrong sft type {self.args.sft_type}!")
        
        self.template = template_dict[self.template_name]
        
        # 下面开始根据 Firefly 中的 chat.py 修改我载入模型的方式
        self.model = ModelUtils.load_model(
            model_name_or_path=self.model_path,
            torch_dtype=torch.float16,
            load_in_4bit=self.args.load_in_4bit,
            load_in_8bit=self.args.load_in_8bit,
            adapter_name_or_path=self.adapter_dir
        ).eval()
        
        # self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).eval()
        # 注意，在 load_in_4bit （或者 8bit）时，这里不需要 .to()
        if not self.args.load_in_4bit and not self.args.load_in_8bit:
            self.model = self.model.to(dtype=torch.float16)
            self.model.to(device)
        
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = load_tokenizer(self.model_path if self.adapter_dir is None else self.adapter_dir)
        
        # 注意，对于 qwen 来讲，stop_word='<|im_end|>'
        if self.template.stop_word is None:
            self.template.stop_word = self.tokenizer.eos_token
        # 获取 stop_word 编码之后的 id
        stop_token_id = self.tokenizer.encode(self.template.stop_word, add_special_tokens=False)
        assert len(stop_token_id) == 1
        self.stop_token_id = stop_token_id[0]
        
        torch.cuda.empty_cache()

    # 接收一定的输入进行输出！
    # 注意，codellama 的话，不可能存在 images 参数！
    # 目前我们只接收单轮对话形式的输入！
    def generate(self, buggy_funcs, systems, datasetname):
        if self.args.do_sample:
            if self.args.do_beam:
                # Beam search
                # the generation stops as soon as there are num_beams complete candidates
                generation_config = GenerationConfig(
                    num_beams=self.args.num_beams,
                    early_stopping=True,
                )
            else:
                # The combination of Top-k & Top-p & Temperature sampling
                generation_config = GenerationConfig(
                    # do_sample=self.args.do_sample,
                    temperature=self.args.temperature if self.args.do_temp else None,
                    top_k=self.args.top_k if self.args.do_topk else None,
                    top_p=self.args.top_p if self.args.do_topp else None,
                )
        
        history = []
        prompt = buggy_funcs[0].strip()
        system = systems[0]
        input_ids = build_prompt(self.tokenizer, self.template, prompt, copy.deepcopy(history), system=system).to(self.model.device)
        
        flag_fail = False
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids, 
                    max_new_tokens=self.args.max_output_len,
                    # 0506 new
                    num_return_sequences=self.args.request_num,
                    do_sample=self.args.do_sample,
                    # top_p=self.args.top_p, 
                    # temperature=self.args.temperature, 
                    # repetition_penalty=self.args.repetition_penalty,
                    eos_token_id=self.stop_token_id,
                    # 0506 new
                    generation_config=generation_config,
                )
        except Exception as e:
            print(f"Exception:{e}")
            flag_fail = True
            # 如果出现了炸显存的情况，那么返回一个空的 outputs
            return [[]]
            # raise Exception("HEY!")
        
        # 处理接收到的 outputs，原始形式应该是一个类似 list 的格式，获取第 0 项之后，只选取 len(input_ids) 后面的部分？[0] 可能是指
        output_ids = outputs[:, len(input_ids[0]):]
        # 对输出的字符串进行解码，由编码形式到 string
        # responses = self.tokenizer.decode(outputs)
        responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # 记录最终处理之后的所有回复
        final_responses = []
        
        # 处理左右两边的空白，并且去除潜在的停止符号，这也是为什么很多回复中会出现 </s>
        for response in responses:
            response = response.strip().replace(self.template.stop_word, "").strip()
            final_responses.append(response)
        
        # 最终返回的应该是一个二层 list
        return [final_responses]
    