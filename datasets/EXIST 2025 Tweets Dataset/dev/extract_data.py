#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os

# 文件路径
input_file = '/t-3058/CLEF/datasets/EXIST 2025 Tweets Dataset/dev/EXIST2025_dev.json'
output_file = '/t-3058/CLEF/datasets/EXIST 2025 Tweets Dataset/dev//EXIST2025_dev_extracted_100.json'

# 读取原始JSON文件
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 创建新的字典来存储提取后的数据
extracted_data = {}
count = 0

# 提取前100条数据的指定字段
for tweet_id, tweet_data in data.items():
    if count >= 100:
        break
    
    extracted_data[tweet_id] = {
        'id_EXIST': tweet_data.get('id_EXIST', ''),
        'lang': tweet_data.get('lang', ''),
        'gender_annotators': tweet_data.get('gender_annotators', []),
        'age_annotators': tweet_data.get('age_annotators', []),
        'ethnicities_annotators': tweet_data.get('ethnicities_annotators', []),
        'study_levels_annotators': tweet_data.get('study_levels_annotators', []),
        'countries_annotators': tweet_data.get('countries_annotators', []),
        'tweet': tweet_data.get('tweet', ''),
        'labels_task1_1': tweet_data.get('labels_task1_1', [])
    }
    count += 1

# 将提取后的数据保存为新的JSON文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(extracted_data, f, ensure_ascii=False, indent=2)

print(f"已成功提取前{count}条数据，并保存到{output_file}文件中。")
print(f"提取的字段: id_EXIST, lang, tweet, labels_task1_1, gender_annotators, age_annotators, ethnicities_annotators, study_levels_annotators, countries_annotators")