import json

in_path = "/data/public/multimodal/lihaoyu/vqa_eval/answers_vqa/answers-0519/minicpm/20240519184718/docVQATest.json"
out_path = "/data/public/multimodal/lihaoyu/vqa_eval/answers_vqa/answers-0519/minicpm/20240519184718/docVQATest_new.json"

with open(in_path , 'r') as f:
    data = json.load(f)
# Convert and simplify the structure
transformed_data = [{"questionId": item["question_id"], "answer": item["answer"].replace("</s>", "")} for item in data]

# Print or use the transformed data
# print(transformed_data)
with open(out_path, 'w') as f:
    json.dump(transformed_data, f)