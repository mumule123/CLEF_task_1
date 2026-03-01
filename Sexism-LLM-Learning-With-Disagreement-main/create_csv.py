#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
from collections import Counter, defaultdict

# 定义数据路径
DATA_DIR = "EXIST 2025 Tweets Dataset"
GOLDS_DIR = "golds"
OUTPUT_DIR = "."  # 输出到当前目录

def count_labels(labels_list):
    """计算标签列表中各类别的出现次数，返回概率分布，确保YES和NO标签都存在"""
    counter = Counter(labels_list)
    total = len(labels_list)
    prob_dist = {label: count/total for label, count in counter.items()}
    
    # 确保YES和NO标签都存在，如果不存在则设置为0
    if 'YES' not in prob_dist:
        prob_dist['YES'] = 0.0
    if 'NO' not in prob_dist:
        prob_dist['NO'] = 0.0
        
    return prob_dist

def process_dataset(dataset_type):
    """处理指定类型的数据集（训练集或开发集）"""
    # 读取原始JSON数据
    file_path = os.path.join(DATA_DIR, dataset_type, f"EXIST2025_{dataset_type}.json")
    print(f"读取数据集: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
        return None
    
    # 处理数据，提取需要的字段
    records = []
    for tweet_id, tweet_data in data.items():
        if 'split' not in tweet_data:  # 跳过不完整的数据
            continue
            
        record = {
            'id_EXIST': tweet_data.get('id_EXIST', tweet_id),
            'lang': tweet_data.get('lang', ''),
            'tweet': tweet_data.get('tweet', ''),
        }
        
        # 处理task1标签（性别歧视识别）
        if 'labels_task1_1' in tweet_data:
            task1_labels = tweet_data['labels_task1_1']
            task1_prob = count_labels(task1_labels)
            record['soft_label_task1'] = str(task1_prob)
        
        records.append(record)
    
    # 创建DataFrame
    df = pd.DataFrame(records)
    
    # 加载gold soft标签（如果存在）替换计算出的软标签
    task = 'task1'  # 只处理task1
    gold_file = os.path.join(GOLDS_DIR, f"EXIST2025_{dataset_type}_{task}_gold_soft.json")
    if os.path.exists(gold_file):
        print(f"使用金标准文件: {gold_file}")
        try:
            with open(gold_file, 'r', encoding='utf-8') as f:
                gold_data = json.load(f)
            
            # 如果gold数据是列表形式
            if isinstance(gold_data, list):
                gold_dict = {item['id']: item['value'] for item in gold_data if 'id' in item and 'value' in item}
                
                # 更新DataFrame中的软标签
                for index, row in df.iterrows():
                    tweet_id = row['id_EXIST']
                    if tweet_id in gold_dict:
                        df.at[index, f'soft_label_{task}'] = str(gold_dict[tweet_id])
        except Exception as e:
            print(f"处理金标准文件时出错: {e}")
    
    return df

def main():
    """主函数"""
    print("开始处理EXIST 2025数据集...")
    
    # 处理训练集
    train_df = process_dataset('training')
    if train_df is not None:
        train_csv_path = os.path.join(OUTPUT_DIR, "EXIST2025_training.csv")
        train_df.to_csv(train_csv_path, index=False)
        print(f"训练集已保存至: {train_csv_path}, 共{len(train_df)}条记录")
    
    # 处理开发集
    dev_df = process_dataset('dev')
    if dev_df is not None:
        dev_csv_path = os.path.join(OUTPUT_DIR, "EXIST2025_dev.csv")
        dev_df.to_csv(dev_csv_path, index=False)
        print(f"开发集已保存至: {dev_csv_path}, 共{len(dev_df)}条记录")
    
    # 合并训练集和开发集
    if train_df is not None and dev_df is not None:
        combined_df = pd.concat([train_df, dev_df])
        combined_csv_path = os.path.join(OUTPUT_DIR, "EXIST2025_training-dev.csv")
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"合并后的训练集和开发集已保存至: {combined_csv_path}, 共{len(combined_df)}条记录")
    
    print("处理完成!")

if __name__ == "__main__":
    main()