#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
提取EXIST 2025数据集中的关键信息，并将其保存为易于使用的CSV格式。

该脚本会从EXIST 2025数据集的training、dev和test集中提取以下信息：
- id_EXIST: 推文ID
- lang: 语言
- tweet: 推文内容
- gender_annotators: 标注者性别
- age_annotators: 标注者年龄
- ethnicities_annotators: 标注者民族背景
- labels_task1_1: 标注任务1的结果

作者: GitHub Copilot
日期: 2025-05-04
"""

import json
import os
import pandas as pd
from tqdm import tqdm

# 定义数据集路径
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "EXIST 2025 Tweets Dataset")
TRAINING_CLEAN = os.path.join(DATASET_DIR, "training", "EXIST2025_training_clean.json")
DEV_CLEAN = os.path.join(DATASET_DIR, "dev", "EXIST2025_dev_clean.json")
TEST_CLEAN = os.path.join(DATASET_DIR, "test", "EXIST2025_test_clean_clean.json")

# 输出文件路径
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
print(OUTPUT_DIR)
OUTPUT_TRAINING = os.path.join(OUTPUT_DIR, "EXIST 2025 Tweets Dataset","training","EXIST2025_training_extracted.csv")
OUTPUT_DEV = os.path.join(OUTPUT_DIR,"EXIST 2025 Tweets Dataset","dev", "EXIST2025_dev_extracted.csv")
OUTPUT_TEST = os.path.join(OUTPUT_DIR,"EXIST 2025 Tweets Dataset","test", "EXIST2025_test_extracted.csv")
OUTPUT_ALL = os.path.join(OUTPUT_DIR, "EXIST2025_all_extracted.csv")

def load_json(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"警告: 文件 {file_path} 不存在")
        return {}
    except json.JSONDecodeError:
        print(f"警告: 文件 {file_path} 不是有效的JSON格式")
        return {}

def extract_information(data):
    """提取所需的信息"""
    extracted_data = []
    
    for tweet_id, tweet_data in tqdm(data.items(), desc="处理数据"):
        # 初始化一个字典来存储当前推文的信息
        tweet_info = {
            'id_EXIST': tweet_data.get('id_EXIST', ''),
            'lang': tweet_data.get('lang', ''),
            'gender_annotators': '',
            'age_annotators': '',
            'ethnicities_annotators': '',
            'tweet': tweet_data.get('tweet', ''),
            'labels_task1_1': ''
        }
        
        # 直接获取性别、年龄和民族背景信息（如果存在）
        # 这些字段在原始数据中可能是数组
        if 'gender_annotators' in tweet_data:
            if isinstance(tweet_data['gender_annotators'], list):
                tweet_info['gender_annotators'] = ','.join(map(str, tweet_data['gender_annotators']))
            else:
                tweet_info['gender_annotators'] = str(tweet_data['gender_annotators'])
        
        if 'age_annotators' in tweet_data:
            if isinstance(tweet_data['age_annotators'], list):
                tweet_info['age_annotators'] = ','.join(map(str, tweet_data['age_annotators']))
            else:
                tweet_info['age_annotators'] = str(tweet_data['age_annotators'])
        
        if 'ethnicities_annotators' in tweet_data:
            if isinstance(tweet_data['ethnicities_annotators'], list):
                tweet_info['ethnicities_annotators'] = ','.join(map(str, tweet_data['ethnicities_annotators']))
            else:
                tweet_info['ethnicities_annotators'] = str(tweet_data['ethnicities_annotators'])
        
        # 处理annotators字段（如果直接存储了annotators对象数组）
        # 这是为了兼容与前面版本不同的数据结构
        if 'annotators' in tweet_data and not 'gender_annotators' in tweet_data:
            # 检查annotators的类型，确保它是一个列表
            if isinstance(tweet_data['annotators'], list):
                # 提取性别信息
                genders = []
                ages = []
                ethnicities = []
                
                for annotator in tweet_data['annotators']:
                    if isinstance(annotator, dict):
                        genders.append(annotator.get('gender', ''))
                        ages.append(str(annotator.get('age', '')))
                        ethnicities.append(annotator.get('ethnicity', ''))
                
                tweet_info['gender_annotators'] = ','.join(genders)
                tweet_info['age_annotators'] = ','.join(ages)
                tweet_info['ethnicities_annotators'] = ','.join(ethnicities)
            elif isinstance(tweet_data['annotators'], str):
                # 如果annotators是字符串，直接使用它
                tweet_info['gender_annotators'] = tweet_data['annotators']
            else:
                # 其他情况，尝试转换为字符串
                try:
                    tweet_info['gender_annotators'] = str(tweet_data['annotators'])
                except:
                    tweet_info['gender_annotators'] = ''
        
        # 提取标注任务1的结果（如果存在）
        if 'labels_task1_1' in tweet_data:
            labels = tweet_data['labels_task1_1']
            if isinstance(labels, list):
                tweet_info['labels_task1_1'] = ','.join(map(str, labels))
            else:
                tweet_info['labels_task1_1'] = str(labels)
        
        # 将信息添加到列表中
        extracted_data.append(tweet_info)
    
    return extracted_data

def save_to_csv(data, output_path):
    """将提取的信息保存为CSV文件"""
    if not data:
        print(f"警告: 没有数据可保存到 {output_path}")
        return
    
    # 确保tweet字段内容被正确引用和转义
    for item in data:
        if 'tweet' in item:
            # 先移除可能已存在的引号以避免重复引用
            tweet_text = item['tweet'].replace('"', '""')  # 将单个引号替换为两个引号(CSV标准转义)
            item['tweet'] = f'"{tweet_text}"'  # 用引号包围整个tweet内容
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, encoding='utf-8', sep='\t', quoting=1)  # quoting=1表示QUOTE_ALL
    print(f"已成功保存数据到 {output_path}")

def process_dataset(input_path, output_path):
    """处理单个数据集"""
    if not os.path.exists(input_path):
        print(f"警告: 数据集文件 {input_path} 不存在")
        return None
    
    print(f"正在处理数据集: {input_path}")
    data = load_json(input_path)
    if not data:
        return None
    
    extracted_data = extract_information(data)
    save_to_csv(extracted_data, output_path)
    return extracted_data

def main():
    """主函数"""
    print("开始提取EXIST 2025数据集信息...")
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 处理训练集
    training_data = process_dataset(TRAINING_CLEAN, OUTPUT_TRAINING)
    
    # 处理开发集
    dev_data = process_dataset(DEV_CLEAN, OUTPUT_DEV)
    
    # 处理测试集
    test_data = process_dataset(TEST_CLEAN, OUTPUT_TEST)
    
    # 合并所有数据（如果都成功处理）
    all_data = []
    if training_data:
        all_data.extend(training_data)
    if dev_data:
        all_data.extend(dev_data)
    
    if all_data:
        save_to_csv(all_data, OUTPUT_ALL)
        print(f"已成功合并所有数据并保存到 {OUTPUT_ALL}")
    
    print("数据提取完成!")

if __name__ == "__main__":
    main()