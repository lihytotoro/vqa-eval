# 将 OCRsft.json 中的回答里的 image 和 answer 进行转存
import os
import json
import shutil
from tqdm import tqdm

# 原始文件
ori_ans_path = "/data/public/multimodal/lihaoyu/vqa_eval/answers-0423/minicpm/20240424002855/OCRsft.json"
# 将所有 samples 的相关内容保存到这个目录下
base_ans_dir = "./ocrsft_ans"

if __name__ == "__main__":
    with open(ori_ans_path, "r") as f:
        samples = json.load(f)
        
    ans_category_cnt = {}
        
    for idx, sample in tqdm(enumerate(samples)):
        src_path = sample["image_path"]
        ans = sample["answer"]
        gt_ans = sample["gt_answers"]
        
        dataset_category = src_path.split("/")[-2]
        img_name = src_path.split("/")[-1]
        if not dataset_category in ans_category_cnt:
            ans_category_cnt[dataset_category] = 0
            subdir_name = f"{dataset_category}_0"
        else:
            c_idx = ans_category_cnt[dataset_category] + 1
            ans_category_cnt[dataset_category] = c_idx
            subdir_name = f"{dataset_category}_{c_idx}"
        save_dir = os.path.join(base_ans_dir, subdir_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        dst_path = os.path.join(save_dir, img_name)
        shutil.copy(src=src_path, dst=dst_path)
        
        with open(os.path.join(save_dir, "ans.txt"), "w", encoding="utf-8") as fa:
            fa.write(ans)
        with open(os.path.join(save_dir, "gt_ans.txt"), "w", encoding="utf-8") as fga:
            fga.write(gt_ans)