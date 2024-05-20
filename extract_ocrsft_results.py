# 从指定路径下获取 jpg 和 txt（从 OCRsft.json 提取），保存
import os
import json
import shutil
from PIL import Image
from tqdm import tqdm

# 在这个目录下读取 json，存储到 extracted 目录下
res_dir = "/data/public/multimodal/lihaoyu/vqa_eval/answers-0424-5/minicpm"
subdir = "extracted"

if __name__ == "__main__":
    with open(os.path.join(res_dir, "OCRsft.json"), "r", encoding="utf-8") as fj:
        res_list = json.load(fj)

    category_cnt_dict = {}

    for res in tqdm(res_list):
        img_path = res["image_path"]
        ans = res["answer"]
        category = img_path.split("/")[-2]
        img_name = img_path.split("/")[-1]
        
        if not category in category_cnt_dict:
            category_cnt_dict[category] = 0
            c_idx = 0
        else:
            c_idx = category_cnt_dict[category] + 1
            category_cnt_dict[category] = c_idx
        
        if not os.path.exists(os.path.join(res_dir, subdir, category + "_" + str(c_idx))):
            os.mkdir(os.path.join(res_dir, subdir, category + "_" + str(c_idx)))
        
        img_dst = os.path.join(res_dir, subdir, category + "_" + str(c_idx), img_name)
        ans_dst = os.path.join(res_dir, subdir, category + "_" + str(c_idx), img_name.removesuffix(".jpg") + ".txt")
        
        shutil.copy(src=img_path, dst=img_dst)
        
        with open(ans_dst, "w", encoding="utf-8") as fa:
            fa.write(ans)