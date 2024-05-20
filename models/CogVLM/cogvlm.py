import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer


class CogVlm():


    def __init__(self,
                 name='cogvlm-chat',
                 tokenizer_name='/data/public/multimodal/multimodal_model_ckpts/vicuna-7b-v1.5',):
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            f'/data/public/multimodal/multimodal_model_ckpts/cogvlm-chat-hf',
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to('cuda').eval()

    def generate(self, images, questions, datasetname=None):
        image_path = images[0]
        # prompt = f"Question: {questions[0]} Answer:"
        prompt = "Answer the question directly with single word" + "\n" + questions[0]
        image = Image.open(image_path).convert('RGB')
        inputs = self.model.build_conversation_input_ids(
            self.tokenizer, query=prompt, history=[], images=[image])  # chat mode
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
        }
        # 在这里设置 beam search 的相关参数
        gen_kwargs = {'max_length': 2048, 'do_sample': False, 'num_beams':3}

        # print("HEYHEYHEY!")
        # print(len(inputs['input_ids']))
        # print(len(inputs['images']))
        # exit()

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            # print(tokenizer.decode(outputs[0]))
            response = self.tokenizer.decode(outputs[0])
        # output = response[len(prompt):]
        return [response]
