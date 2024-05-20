import os
import sys
import torch
from tqdm import tqdm
from datasets.vqa_dataset import GroundingDataset

grounding_dataset_dir = "/home/lihaoyu/code/0411/valid_samples_bbox_all_red"
grounding_dataset_dir_type = "layers"

# sample_start_idx = 7219
# sample_end_idx = 7220

dataset = GroundingDataset(bbox_dir=grounding_dataset_dir, bbox_dir_type=grounding_dataset_dir_type)
# dataset = torch.utils.data.Subset(dataset, range(sample_start_idx, sample_end_idx))

print("finish constructing dataset!")

image_paths = []

for item in tqdm(dataset):
    image_paths.append(item["image_path"])
    # print(item["image_path"])
    
print(len(image_paths))

# target_paths = ["/home/lihaoyu/code/0411/valid_samples_bbox_all_red/save_V0403/0405/17/df_ver=save_V0403_day=0405_hour=17_valid_idx=2-867.png",
#                 "/home/lihaoyu/code/0411/valid_samples_bbox_all_red/save_V0403/0405/02/df_ver=save_V0403_day=0405_hour=02_valid_idx=2-702.png",
#                 "/home/lihaoyu/code/0411/valid_samples_bbox_all_red/save_V0403/0405/06/df_ver=save_V0403_day=0405_hour=06_valid_idx=11-185.png",
#                 "/home/lihaoyu/code/0411/valid_samples_bbox_all_red/save_V0403/0405/12/df_ver=save_V0403_day=0405_hour=12_valid_idx=14-356.png",
#                 "/home/lihaoyu/code/0411/valid_samples_bbox_all_red/save_V0403/0402/23/df_ver=save_V0403_day=0402_hour=23_valid_idx=9-257.png",
#                 "/home/lihaoyu/code/0411/valid_samples_bbox_all_red/save_V0403/0402/18/df_ver=save_V0403_day=0402_hour=18_valid_idx=0-203.png",
#                 "/home/lihaoyu/code/0411/valid_samples_bbox_all_red/save_V0403/0403/23/df_ver=save_V0403_day=0403_hour=23_valid_idx=4-373.png",
#                 "/home/lihaoyu/code/0411/valid_samples_bbox_all_red/save_V0403/0403/22/df_ver=save_V0403_day=0403_hour=22_valid_idx=0-212.png"]

target_paths = ["/home/lihaoyu/code/0411/valid_samples_bbox_all_red/save_V0403/0405/17/df_ver=save_V0403_day=0405_hour=17_valid_idx=4-954.png",
                "/home/lihaoyu/code/0411/valid_samples_bbox_all_red/save_V0403/0405/02/df_ver=save_V0403_day=0405_hour=02_valid_idx=1-444.png",
                "/home/lihaoyu/code/0411/valid_samples_bbox_all_red/save_V0403/0405/06/df_ver=save_V0403_day=0405_hour=06_valid_idx=2-814.png",
                "/home/lihaoyu/code/0411/valid_samples_bbox_all_red/save_V0403/0405/12/df_ver=save_V0403_day=0405_hour=12_valid_idx=6-504.png",
                "/home/lihaoyu/code/0411/valid_samples_bbox_all_red/save_V0403/0402/23/df_ver=save_V0403_day=0402_hour=23_valid_idx=9-687.png",
                "/home/lihaoyu/code/0411/valid_samples_bbox_all_red/save_V0403/0402/18/df_ver=save_V0403_day=0402_hour=18_valid_idx=7-843.png",
                "/home/lihaoyu/code/0411/valid_samples_bbox_all_red/save_V0403/0403/23/df_ver=save_V0403_day=0403_hour=23_valid_idx=4-440.png",
                "/home/lihaoyu/code/0411/valid_samples_bbox_all_red/save_V0403/0403/22/df_ver=save_V0403_day=0403_hour=22_valid_idx=4-135.png"]

for target_path in target_paths:
    print(f"rank:{target_paths.index(target_path)}\tindex:{image_paths.index(target_path)}")