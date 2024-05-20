# 将 0415 跑完的所有 grounding 数据合并
import os
import json
from tqdm import tqdm

base_data_dir_1 = "/data/public/multimodal/lihaoyu/vqa_eval/answers-new/minicpm/0415-grounding"
base_data_dir = "/data/public/multimodal/lihaoyu/vqa_eval/answers-new/minicpm"
data_dirs_1 = [os.path.join(base_data_dir_1, data_subdir) for data_subdir in os.listdir(base_data_dir_1)]
data_dirs_2 = [os.path.join(base_data_dir, data_subdir) for data_subdir in os.listdir(base_data_dir) if data_subdir.startswith("0415-grounding-recover-")]

if __name__ == "__main__":
    assert len(data_dirs_1) == 8
    assert len(data_dirs_2) == 8
    
    # 获取二级目录，拼接
    data_files_2 = [os.path.join(data_dir_2, data_subdir, "Grounding.json") for data_dir_2 in data_dirs_2 for data_subdir in os.listdir(data_dir_2) if data_subdir.endswith("rank=0")]
    
    assert len(data_files_2) == 8
    
    # 存储所有 json
    merged_data = []
    all_image_paths = []
    
    # 首先处理原始目录
    for data_dir_1 in tqdm(data_dirs_1):
        js_list = [os.path.join(data_dir_1, js_file) for js_file in os.listdir(data_dir_1)]
        for js in js_list:
            with open(js, "r", encoding="utf-8") as f:
                tmp_data = json.load(f)
                for item in tmp_data:
                    img_path = item["image_path"]
                    all_image_paths.append(img_path)
                merged_data.extend(tmp_data)
                
    
    # 168000
    print(len(merged_data))
    
    # 处理 recover 目录
    for data_file in tqdm(data_files_2):
        with open(data_file, "r", encoding="utf-8") as f:
            tmp_data = json.load(f)
            for item in tmp_data:
                img_path = item["image_path"]
                all_image_paths.append(img_path)
            merged_data.extend(tmp_data)
    
    all_image_paths = list(set(all_image_paths))
    
    # 668511? 去重之后发现没有重复的
    print(len(merged_data))
    print(len(all_image_paths))
    print(all_image_paths[0:10])
    
    # # 向这里写入了答案？
    # with open("/home/lihaoyu/code/0413/Grounding.json", "w", encoding="utf-8") as f:
    #     json.dump(merged_data, f)