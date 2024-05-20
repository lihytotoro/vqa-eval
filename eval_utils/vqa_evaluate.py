import itertools
import json
import os
import re
from collections import Counter, namedtuple

import sacrebleu
import torch
from tqdm import tqdm


class InferenceSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, size):
        self._size = int(size)
        assert size > 0
        self._rank = torch.distributed.get_rank()
        self._world_size = torch.distributed.get_world_size()
        self._local_indices = self._get_local_indices(size, self._world_size,
                                                      self._rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]

        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[:rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)
    
def collate_fn_vqa(batches):
    '''
    '''
    image_paths = [_['image_path'] for _ in batches]
    questions = [_['question'] for _ in batches]
    gt_answers = [_['gt_answers'] for _ in batches]
    ocr_tokens = [_['ocr_tokens'] if 'ocr_tokens' in _ else None for _ in batches]
    question_ids = [_['question_id'] if 'question_id' in _ else None for _ in batches]
    question_type = [_['question_type'] if 'question_type' in _ else None for _ in batches]

    return image_paths, questions, gt_answers, ocr_tokens, question_ids, question_type


def collate_fn_apr(batches):
    '''
    处理 APR 数据集（主要是 defects4j）需要进行的预处理
    '''
    bug_ids = [item['bug_id'] for item in batches]
    inputs = [item['input'] for item in batches]
    gt_outputs = [item['gt_output'] for item in batches]
    
    return bug_ids, inputs, gt_outputs


# 这个函数主要用于处理 cwe-inference 任务对应的数据集
def collate_fn_cweinf(batches):
    '''
    bug_ids? 注意，java-juliet 数据集没有这个输入项，但是可以进行认为编号？
    inputs: 仍然代表输入的 buggy code
    gt_cwes: 对于 java-juliet 数据集，有 ground truth，但是对于 d4j 数据集则没有！
    '''
    bug_ids = [item['bug_id'] for item in batches]
    systems = [item['system'] for item in batches]
    inputs = [item['input'] for item in batches]
    gt_cwes = [item['gt_cwe'] for item in batches]
    
    return bug_ids, systems, inputs, gt_cwes


def has_word(sentence, word):
    if word[0].isalnum():
        start_pattern = r"\b"
    else:
        start_pattern = r""

    if word[-1].isalnum():
        end_pattern = r"\b"
    else:
        end_pattern = r""

    pattern = start_pattern + re.escape(word) + end_pattern
    match = re.search(pattern, sentence)
    return bool(match)

def remove_special_chars(s):
    pattern = r"[^a-zA-Z0-9\s]"
    s = re.sub(pattern, "", s)
    return s

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]
class VQAEval:
    def __init__(self):
        self.contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
        }
        self.manualMap = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        self.articles = ["a", "an", "the"]

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(\,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]
    def clean_text(self, text):
        text = text.replace("\n", " ").replace("\t", " ").strip()
        text = self.processPunctuation(text)
        text = self.processDigitArticle(text)
        return text
    
    def evaluate_has(self, answer, gt_answers):
        # 最不严谨但是最能反映实际情况的方法
        answer = answer.replace("\n", " ")
        answer = answer.replace("\t", " ")
        answer = answer.strip()
        answer = self.processPunctuation(answer)
        answer = self.processDigitArticle(answer)
        if type(gt_answers)==list:
            for i in range(len(gt_answers)):
                gt_answers[i] = gt_answers[i].replace("\n", " ")
                gt_answers[i] = gt_answers[i].replace("\t", " ")
                gt_answers[i] = gt_answers[i].strip()
                gt_answers[i] = self.processPunctuation(gt_answers[i])
                gt_answers[i] = self.processDigitArticle(gt_answers[i])
                if has_word(answer, gt_answers[i]):
                    return 1
            return 0
        else:
            gt_answers = gt_answers.replace("\n", " ")
            gt_answers= gt_answers.replace("\t", " ")
            gt_answers = gt_answers.strip()
            gt_answers = self.processPunctuation(gt_answers)
            gt_answers = self.processDigitArticle(gt_answers)
            if has_word(answer, gt_answers):
                return 1
            else:
                return 0

    def evaluate_bleu(self, candidate, references):
        # 应该要换成pycoco_bleu
        # TODO： 加入Ciyder？
        sys = [self.clean_text(candidate)]

        if isinstance(references, str):
            refs = [[self.clean_text(references)]]
        else: 
            refs = [[self.clean_text(ref) for ref in references]]
        bleu = sacrebleu.corpus_bleu(sys, refs)
        return bleu.score        
    
    def evaluate_precision(self, answer, gt_answers):
        # TODO： 等待检查
        def clean_and_tokenize(text):
            text = self.clean_text(text)
            return re.findall(r'\w+', text)

        answer_words = clean_and_tokenize(answer)

        if isinstance(gt_answers, list):
            gt_words = []
            for gt in gt_answers:
                gt_words.extend(clean_and_tokenize(gt))
        else:
            gt_words = clean_and_tokenize(gt_answers)

        gt_counter = Counter(gt_words)
        tp = 0  # true positive

        for word in answer_words:
            if gt_counter[word] > 0:
                tp += 1
                gt_counter[word] -= 1

        fp = len(answer_words) - tp  # false positive

        return tp / (tp + fp) if tp + fp > 0 else 0
    def evaluate_precision_word(self, answer, gt_answers):
        # TODO： 等待检查
        answer = self.clean_text(answer)
        
        if isinstance(gt_answers, list):
            gt_answers = [self.clean_text(gt) for gt in gt_answers]
            combined_gt = "".join(gt_answers)
        else:
            combined_gt = self.clean_text(gt_answers)
        
        gt_counter = Counter(combined_gt.split())
        tp = 0
        
        for word in answer.split():
            if gt_counter[word] > 0:  
                tp += 1  
                gt_counter[word] -= 1 
        
        fp = len(answer.split()) - tp  

        return tp / (tp + fp) if tp + fp > 0 else 0
    
    def evaluate_recall_word(self, answer, gt_answers):
        # TODO： 等待检查
        answer = self.clean_text(answer)

        if isinstance(gt_answers, list):
            gt_answers = [self.clean_text(gt) for gt in gt_answers]
            combined_gt = "".join(gt_answers)
        else:
            combined_gt = self.clean_text(gt_answers)

        gt_counter = Counter(combined_gt.split())
        tp = 0
        total_gt = sum(gt_counter.values())  # Total number of words in ground truth

        for word in answer.split():
            if gt_counter[word] > 0:
                tp += 1
                gt_counter[word] -= 1

        return tp / total_gt if total_gt > 0 else 0
    
    def evaluate_f_score_word(self, answer, gt_answers):
        # TODO： 等待检查
        precision = self.evaluate_precision_word(answer, gt_answers)
        recall = self.evaluate_recall_word(answer, gt_answers)

        if precision + recall > 0:
            return 2 * (precision * recall) / (precision + recall)
        else:
            return 0


    # def evaluate_precision(self, answer, gt_answers):
    #     '''FUNSD'''
    #     answer = self.clean_text(answer)
        
    #     if type(gt_answers) == list:
    #         gt_answers = [self.clean_text(gt) for gt in gt_answers]
    #         combined_gt = "".join(gt_answers)
    #     else:
    #         combined_gt = self.clean_text(gt_answers)

    #     tp = sum([1 for char in answer if char in combined_gt])
    
    #     fp = len(answer) - tp

    #     if tp + fp == 0:
    #         return 0
    #     else:
    #         return tp / (tp + fp)
    
    def evaluate_vqa_human(self, answer, gt_answers):
        '''TextVQA, VQAv2, OKVQA, vizwiz'''
        answer = answer.replace("\n", " ").replace("\t", " ").strip()
        answer = self.processPunctuation(answer)
        answer = self.processDigitArticle(answer)
        gt_answers = [self.processPunctuation(ans) for ans in gt_answers]
        gt_answers = [self.processDigitArticle(ans) for ans in gt_answers]

        gtAcc = [] 

        for idx, gtAnsDatum in enumerate(gt_answers):  
            otherGTAns = gt_answers[:idx] + gt_answers[idx+1:]

            matchingAns = [item for item in otherGTAns if answer == item] 
            # matchingAns = [item for item in otherGTAns if has_word(answer, item)] 
            
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc) 

        avgGTAcc = float(sum(gtAcc)) / len(gtAcc) if gtAcc else 0  

        return avgGTAcc  
    
    def evaluate_em(self, answer, gt_answers):
        '''ocrvqa, AI2D, TVQA, RefExp'''
        answer = answer.lower().replace("\n", " ").replace("\t", " ").strip()
        
        if isinstance(gt_answers, str):
            gt_answers = [gt_answers]
        
        gt_answers = [ans.lower().replace("\n", " ").replace("\t", " ").strip() for ans in gt_answers]
        
        exact_match_scores = [(1.0 if answer == gt_ans else 0.0) for gt_ans in gt_answers]
        
        return 1 if max(exact_match_scores) == 1.0 else 0

    def evaluate_anls(self, answer, gt_answers, threshold=0.5):
        '''DOcVQA, InfographicsVQA, STVQA'''
        # TODO： 精炼的infographicseval.py 很久没有检查了
        answer = ' '.join(answer.strip().lower().split())
        if not isinstance(gt_answers, list):
            gt_answers = [gt_answers]
        gt_answers = [' '.join(gt_answer.strip().lower().split()) for gt_answer in gt_answers]

        values = []
        for gt_answer in gt_answers:
            dist = levenshtein_distance(answer, gt_answer)
            length = max(len(answer), len(gt_answer))
            values.append(0.0 if length == 0 else float(dist) / float(length))

        score = 1 - min(values)
        
        score = 0 if score < threshold else score
        
        return score
    
    def evaluate_NED(self, answer, gt_answer):
        # 继承自 kosmos2.5 的公示
        print(f"answer:{answer}")
        print(f"gt_answer:{gt_answer}")
        answer = ' '.join(answer.strip().lower().split())
        gt_answer = ' '.join(gt_answer.strip().lower().split())

        max_length = max(len(answer), len(gt_answer))

        if max_length == 0:
            return 0.0

        actual_distance = levenshtein_distance(answer, gt_answer)

        ned = actual_distance / max_length
        return (1-ned)*100

    def evaluate_IoU(self,answer_box, gt_box):
        # TODO： 等待检查， GPT4生成的0.0
        answer_x1, answer_y1, answer_x2, answer_y2 = answer_box
        gt_x1, gt_y1, gt_x2, gt_y2 = gt_box

        x1 = max(answer_x1, gt_x1)
        y1 = max(answer_y1, gt_y1)
        x2 = min(answer_x2, gt_x2)
        y2 = min(answer_y2, gt_y2)

        if x2 < x1 or y2 < y1:
            # print(f"Invalid box: {answer_box}, {gt_box}")
            return 0.0

        intersection_area = (x2 - x1) * (y2 - y1)

        answer_area = (answer_x2 - answer_x1) * (answer_y2 - answer_y1)
        gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)

        iou = intersection_area / float(answer_area + gt_area - intersection_area)
        return iou * 100 

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) != None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = " ".join(outText)
        return outText
    
def evaluate_dataset(dataset_name, answer_file_path, model_name, method = None):
    with open(answer_file_path, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    eval = VQAEval()
    total_accuracy = 0
    num = 0
    Entry = namedtuple('Entry', ['text', 'bbox'])

    for item in predictions:
        gt_answers = item['gt_answers']
        answer = item['answer']
        if method is not None:
            # 我实际上没有传这个参数
            # case : vqa\em\anls\precision\ned\iou\bleu
            pass
        if dataset_name in ["textVQA", "OKVQA", "vizwiz",'VQAv2']:
            if num == 0:
                print(f"evaluating vqa...")
            accuracy = eval.evaluate_vqa_human(answer, gt_answers)
            # accuracy = eval.evaluate_vqa(answer, gt_answers)
        elif dataset_name in ['ocrVQA', 'WebSRC']:
            if num == 0:
                print(f"evaluating em...")
            accuracy = eval.evaluate_em(answer, gt_answers)
        elif dataset_name in ['docVQA', 'STVQA', 'InfographicsVQA']:
            if  num == 0:
                print(f"evaluating anls...")
            accuracy = eval.evaluate_anls(answer, gt_answers)
        elif dataset_name in ['FUNSD']:
            if num == 0:
                print(f"evaluating precision...")
            accuracy = eval.evaluate_precision(answer, gt_answers)
        elif dataset_name in ['C4WEB']:
            if num == 0:
                print(f"evaluating NED...")
            answer = re.findall(r'<ref>(.*?)</ref>', answer)
            answer = ' '.join(answer).strip()
            gt_answers = re.sub(r'\s+', ' ', gt_answers)
            gt_answers = re.findall(r'<ref>(.*?)</ref>', gt_answers)
            gt_answers = ' '.join(gt_answers).strip()
            accuracy = eval.evaluate_NED(answer, gt_answers)
            print(f"NED:{accuracy}")
        elif dataset_name in ['IDLOCR']:
            if num == 0:
                print(f"evaluating NED...")
            answer = re.findall(r'<ref>(.*?)</ref>', answer)
            answer = ' '.join(answer).strip()
            gt_answers = re.sub(r'\s+', ' ', gt_answers)
            gt_answers = re.findall(r'<ref>(.*?)</ref>', gt_answers)
            gt_answers = ' '.join(gt_answers).strip()
            accuracy = eval.evaluate_NED(answer, gt_answers)
            print(f"NED:{accuracy}")
        # elif dataset_name in ['WebC4']:
        # 这是旧版WebC4的评估方法
        #     answer_entries = [Entry(text, bbox) for text, bbox in zip(item['answer_texts'], item['answer_boxes'])]
        #     gt_entries = [Entry(text, bbox) for text, bbox in zip(item['gt_answer_texts'], item['gt_answer_boxes'])]
        #     if method == 'NED':
        #         if num == 0:
        #             print(f"evaluating ned...")
        #         answer_text = ' '.join([entry.text for entry in answer_entries])
        #         gt_text = ' '.join([entry.text for entry in gt_entries])
        #         accuracy = eval.evaluate_NED(answer_text, gt_text)
        #     elif method == 'IoU':
        #         if num == 0:
        #             print(f"evaluating iou...")
        #         ious = []
        #         for answer_entry in answer_entries:
        #             for gt_entry in gt_entries:
        #                 ned = eval.evaluate_NED(answer_entry.text, gt_entry.text)
        #                 if ned >= 50:
        #                     ious.append(eval.evaluate_IoU(answer_entry.bbox, gt_entry.bbox))
        #         accuracy = sum(ious) / len(ious) if ious else 0
        elif dataset_name in ['VisualMRC']:
            if num == 0:
                print(f"evaluating bleu...")
            accuracy = eval.evaluate_bleu(answer, gt_answers)

        else:
            accuracy = eval.evaluate_has(answer, gt_answers)
        item['accuracy'] = accuracy

        total_accuracy += accuracy
        num += 1

    average_accuracy = total_accuracy / num
    print(f'{dataset_name}:{average_accuracy}')
    
    answer_model_method_path = answer_file_path.replace('.json', f'_{model_name}_{method}.json')
    with open(answer_model_method_path, "w", encoding='utf-8') as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)

    return average_accuracy


# 这个是测试时调用的函数
# 最终大部分情况都要返回一个 acc
# 但是，针对 MathVista 和 MMVet 数据集的 case study 中，只需要分析小部分测例，也自然没有 acc 的计算了
# 注意，这里进行 eval 的时候，默认都是针对有图的数据集！
def evaluate_VQA(
    model,
    dataset,
    model_name,
    dataset_name,
    time,
    batch_size=1,
    generate_method="old",
    answer_path='./answers',
    day_subdir_path="",
    save_in_progress=False
):
    print(f"answer path:{answer_path}")

    sampler = None
    if torch.distributed.is_initialized():
        # sampler = DistributedSampler(dataset)
        sampler=InferenceSampler(len(dataset))

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn_vqa
    )
    
    now_rank = torch.distributed.get_rank()
    if day_subdir_path is None or day_subdir_path == "":
        if dataset_name == "Grounding":
            answer_dir = os.path.join(answer_path, model_name, f"grounding_time={time}_rank={now_rank}")
        else:
            answer_dir = os.path.join(answer_path, model_name, time)
    else:
        if dataset_name == "Grounding":
            answer_dir = os.path.join(answer_path, model_name, day_subdir_path, f"grounding_time={time}_rank={now_rank}")
        else:
            answer_dir = os.path.join(answer_path, model_name, day_subdir_path)
    os.makedirs(answer_dir, exist_ok=True)
    
    image_list = []
    for item in dataset:
        image_list.append(item["image_path"])
        # if now_rank == 0:
        #     print(item["image_path"])
    
    # 存储所有 sample 的推理结果
    predictions = []
    # 满 1000 个 answer 保存一次
    if save_in_progress:
        predictions_1k = []
        save_1k_idx = 0
    
    # print(f"rank:{now_rank}")
    # for batch in tqdm(dataloader, desc="Running inference"):
    #     image_paths, questions, gt_answers, ocr_tokens_list, question_ids, question_type  = batch
    #     for idx in range(len(image_paths)):
    #         print(f"rank{now_rank}:{image_paths}\t{image_list.index(image_paths[idx])}")
    # exit()
    
    # 这里开始进行每张卡下的 inference
    for batch in tqdm(dataloader, desc="Running inference"):
        # 从这里获取 batch 的信息！
        image_paths, questions, gt_answers, ocr_tokens_list, question_ids, question_type  = batch

        # print(len(image_paths))

        with torch.no_grad():
            # 这里，使用 model 生成回复
            # 注意，interleaved 是单独针对 minicpm 而言的！
            
            if model_name != "minicpm":
                if model_name != "codellama":
                    outputs = model.generate(images=image_paths, questions=questions, datasetname=dataset_name)
                else:
                    # 0513 new: codellama 的输出与其他的数据集区别很大！
                    outputs = model.generate()
            elif model_name == "minicpm":
                if generate_method == "old":
                    outputs = model.generate(images=image_paths, questions=questions, datasetname=dataset_name)
                elif generate_method == "interleave":
                    outputs = model.generate_with_interleaved(images=image_paths, questions=questions, datasetname=dataset_name)
                else:
                    raise Exception(f"Wrong generate paradigm {generate_method}!")
            
            # outputs 是一个回复的列表，因此逐个获取回答
            # 如果出现了 outputs 为空的情况，那么 predictions 和 predictions_1k 都不会受影响
            for i in range(len(outputs)):
                # 这里是回答构造的格式
                # 注意这里也许可以进行调整
                if dataset_name != "LLaVABenchMultilingual":
                    answer_dict = {
                        'question_id': question_ids[i],
                        'question': questions[i],
                        'answer': outputs[i],
                        'gt_answers': gt_answers[i],
                        'image_path': image_paths[i],
                        'model_name': model_name,
                        'question_type': question_type[i]
                    }
                else:
                    image_name = image_paths[i].split("/")[-1].removesuffix(".jpg")
                    lang = image_name.split("_")[0]
                    sample_idx = int(image_name.split("_")[1])
                    answer_dict = {
                        'lang': lang,
                        'sample_idx': sample_idx,
                        'question_id': question_ids[i],
                        'question': questions[i],
                        'answer': outputs[i],
                        'gt_answers': gt_answers[i],
                        'image_path': image_paths[i],
                        'model_name': model_name,
                        'question_type': question_type[i]
                    }                    
                predictions.append(answer_dict)
                if save_in_progress:
                    predictions_1k.append(answer_dict)
                    if len(predictions_1k) > 999:
                        # 写到指定目录下，进程之间互不影响，独立计数
                        save_file_name = f"{dataset_name}_rank={str(now_rank)}_idx={str(save_1k_idx)}.json"
                        answer_1k_file_path = os.path.join(answer_dir, save_file_name)
                        # 将满编的文件保存到指定路径下
                        with open(answer_1k_file_path, "w", encoding='utf-8') as f:
                            json.dump(predictions_1k, f, indent=4, ensure_ascii=False)
                        predictions_1k.clear()
                        save_1k_idx += 1
    
    # 0415 new: 这里有一个小漏洞——最后一批存在 predictions_1k 里面的数据不会被写入文件中！
    if save_in_progress:
        if len(predictions_1k) > 0:
            # 写到指定目录下，进程之间互不影响，独立计数
            save_file_name = f"{dataset_name}_rank={str(now_rank)}_idx={str(save_1k_idx)}.json"
            answer_1k_file_path = os.path.join(answer_dir, save_file_name)
            # 将满编的文件保存到指定路径下
            with open(answer_1k_file_path, "w", encoding='utf-8') as f:
                json.dump(predictions_1k, f, indent=4, ensure_ascii=False)
            predictions_1k.clear()
            save_1k_idx += 1
                    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        merged_predictions = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_predictions, predictions)
        predictions = [_ for _ in itertools.chain.from_iterable(merged_predictions)]

    # 对于非主进程，直接返回 None
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return None
    
    answer_file_path = os.path.join(answer_dir, f"{dataset_name}.json")
    print(f"answer_file_path:{answer_file_path}")
    
    with open(answer_file_path, "w", encoding='utf-8') as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)

    # 0329 new: 如果 dataset 是 MathVista 或者 MMVet，那么不需要返回 acc
    # 实际上，如果是 d4j 进行推理的话，也不需要进行 acc 的生成
    if dataset_name in ["MathVista", "MMVet", "docVQATest", "objhal", "Grounding", "OCRsft", "OCRcode", "LLaVABenchMultilingual"]:
        return -1.0

    return evaluate_dataset(answer_file_path=answer_file_path, dataset_name=dataset_name, model_name=model_name)


# 0513 new
# 将 evaluate_VQA 进行调整，创建一个专门用于测试
def evaluate_APR(
    model,
    dataset,
    model_name,
    dataset_name,
    batch_size=1,
    output_path="",
):
    # 理论上应该是 483？
    assert len(dataset) == 483

    sampler = None
    if torch.distributed.is_initialized():
        # sampler = DistributedSampler(dataset)
        sampler=InferenceSampler(len(dataset))

    # 获取数据集，理论上测试时 batchsize 固定为 1
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn_apr
    )
    
    now_rank = torch.distributed.get_rank()
    
    output_file_name = output_path.split("/")[-1]
    output_dir = output_path.removesuffix("/" + output_file_name)
    
    # 这里存储预测的结果？
    predictions = []
    
    for batch in tqdm(dataloader, desc="Running inference"):
        # 从这里获取 batch 的信息！
        bug_ids, buggy_inputs, gt_outputs = batch

        assert type(bug_ids) == list
        assert type(buggy_inputs) == list
        assert type(gt_outputs) == list

        with torch.no_grad():
            # 这里，使用 model 生成回复
            # 注意，interleaved 是单独针对 minicpm 而言的！
            
            # 0513 new: codellama
            outputs = model.generate(buggy_funcs=buggy_inputs, datasetname=dataset_name)
            
            assert len(outputs) == 1
            # assert len(outputs[0]) == 10
            
            if len(outputs[0]) == 0:
                answer_dict = {
                    'bug_id': bug_ids[0],
                    'input': buggy_inputs[0],
                    # 这表示对该 sample 进行输出的过程中遇到了显存问题
                    'output_patches': [],
                    'gt_outputs': gt_outputs[0],
                    'model_name': model_name,
                    'dataset_name': dataset_name,
                }
                predictions.append(answer_dict)
            else:
                # outputs 是一个回复的列表，因此逐个获取回答
                for i in range(len(outputs)):
                    # 这里是回答构造的格式
                    answer_dict = {
                        'bug_id': bug_ids[i],
                        'input': buggy_inputs[i],
                        'output_patches': outputs[i],
                        'gt_outputs': gt_outputs[i],
                        'model_name': model_name,
                        'dataset_name': dataset_name,
                    }
                    predictions.append(answer_dict)
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        merged_predictions = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_predictions, predictions)
        predictions = [_ for _ in itertools.chain.from_iterable(merged_predictions)]

    # 对于非主进程，直接返回 None
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return None
    
    # 这里，我们需要修正一下输出的文件的路径和文件名
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)

    if dataset_name in ["Defects4J"]:
        return -1.0

    raise Exception("While inference d4j dataset, we should not reach here!")

    return evaluate_dataset(answer_file_path=output_dir, dataset_name=dataset_name, model_name=model_name)


# 0515 new
# 将 evaluate_VQA 进行调整，创建一个专门用于测试 CWE 推理的类
def evaluate_CWEINF(
    model,
    dataset,
    model_name,
    dataset_name,
    batch_size=1,
    output_path="",
):
    # 理论上应该是 483？
    print(f"len cwe dataset:{len(dataset)}")

    sampler = None
    if torch.distributed.is_initialized():
        # sampler = DistributedSampler(dataset)
        sampler=InferenceSampler(len(dataset))

    # 获取数据集，理论上测试时 batchsize 固定为 1
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn_cweinf
    )
    
    now_rank = torch.distributed.get_rank()
    
    output_file_name = output_path.split("/")[-1]
    output_dir = output_path.removesuffix("/" + output_file_name)
    
    # 这里存储预测的结果？
    predictions = []
    
    for batch in tqdm(dataloader, desc="Running inference"):
        # 从这里获取 batch 的信息！
        bug_ids, systems, buggy_inputs, gt_cwes = batch

        assert type(bug_ids) == list
        assert type(systems) == list
        assert type(buggy_inputs) == list
        assert type(gt_cwes) == list

        with torch.no_grad():
            # 这里，使用 model 生成回复
            # 注意，interleaved 是单独针对 minicpm 而言的！
            
            # 0515 new: qwen
            # 注意，这里的 generate 可能会有细微的差别？
            outputs = model.generate(buggy_funcs=buggy_inputs, systems=systems, datasetname=dataset_name)
            
            assert len(outputs) == 1
            # assert len(outputs[0]) == 10
            
            if len(outputs[0]) == 0:
                answer_dict = {
                    'bug_id': bug_ids[0],
                    'system': systems[0], 
                    'input': buggy_inputs[0],
                    # 这表示对该 sample 进行输出的过程中遇到了显存问题
                    'output_cwe_patches': [],
                    'gt_cwes': gt_cwes[0],
                    'model_name': model_name,
                    'dataset_name': dataset_name,
                }
                predictions.append(answer_dict)
            else:
                # outputs 是一个回复的列表，因此逐个获取回答
                for i in range(len(outputs)):
                    # 这里是回答构造的格式
                    answer_dict = {
                        'bug_id': bug_ids[i],
                        'system': systems[i], 
                        'input': buggy_inputs[i],
                        'output_cwe_patches': outputs[i],
                        'gt_cwes': gt_cwes[i],
                        'model_name': model_name,
                        'dataset_name': dataset_name,
                    }
                    predictions.append(answer_dict)
    
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        merged_predictions = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(merged_predictions, predictions)
        predictions = [_ for _ in itertools.chain.from_iterable(merged_predictions)]

    # 对于非主进程，直接返回 None
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return None
    
    # 这里，我们需要修正一下输出的文件的路径和文件名
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(predictions, f, indent=4, ensure_ascii=False)

    if dataset_name in ["Defects4J", "Java-Juliet"]:
        return -1.0

    raise Exception("While inference d4j dataset, we should not reach here!")