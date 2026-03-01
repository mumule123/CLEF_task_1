#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXIST 2025 Tweets Dataset Cleaner
清洗推文数据，移除用户名、URL、表情符号、标签等，并将文本转换为小写。
"""

import json
import re
import os
import emoji
import argparse
from tqdm import tqdm

def clean_tweet(tweet):

    # 如果推文为空或None，返回空字符串
    if not tweet or tweet is None:
        return ""
    
    # 移除用户名（以@开头的内容）
    tweet = re.sub(r'@\w+', '', tweet)
    
    # 移除URL
    tweet = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tweet)
    tweet = re.sub(r'www\.[^\s]+', '', tweet)
    
    # 移除标签（以#开头的内容）
    tweet = re.sub(r'#\w+', '', tweet)
    
    # 移除表情符号
    tweet = emoji.replace_emoji(tweet, replace='')
    
    # 移除RT前缀
    tweet = re.sub(r'^RT\s+', '', tweet)
    
    # 移除额外的空格、换行符等
    tweet = re.sub(r'\s+', ' ', tweet)
    
    
    # 将文本转换为小写
    tweet = tweet.lower()
    
    # 去除首尾空格
    tweet = tweet.strip()
    
    return tweet

def clean_dataset(input_file, output_file):
    """
    清洗整个数据集的推文。
    
    Args:
        input_file (str): 输入JSON文件路径
        output_file (str): 输出JSON文件路径
    """
    try:
        # 加载原始数据
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"正在清洗数据集: {input_file}")
        print(f"总计 {len(data)} 条推文...")
        
        # 创建一个新的字典用于存储清洗后的数据
        cleaned_data = {}
        
        # 遍历每条推文并清洗
        for tweet_id, tweet_data in tqdm(data.items(), desc="清洗推文"):
            # 复制原始数据
            cleaned_tweet_data = tweet_data.copy()
            
            # 如果存在推文字段，清洗它
            if 'tweet' in cleaned_tweet_data:
                cleaned_tweet_data['tweet'] = clean_tweet(cleaned_tweet_data['tweet'])
                
                # 添加原始推文字段以便参考
                cleaned_tweet_data['original_tweet'] = tweet_data['tweet']
            
            # 将清洗后的推文数据添加到新字典中
            cleaned_data[tweet_id] = cleaned_tweet_data
        
        # 保存清洗后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        
        print(f"清洗完成，已保存到: {output_file}")
        
    except Exception as e:
        print(f"错误: {e}")

def main():
    """
    主函数，处理命令行参数并执行数据清洗。
    """
    parser = argparse.ArgumentParser(description='清洗EXIST 2025 Tweets数据集')
    parser.add_argument('--input', '-i', type=str, help='输入JSON文件路径',)
    parser.add_argument('--output', '-o', type=str, help='输出JSON文件路径')
    parser.add_argument('--all', '-a', action='store_true', help='清洗所有数据集（训练集、开发集和测试集）')
    
    args = parser.parse_args()
    
    # 定义固定的数据集目录
    base_dir = '/t-3058/CLEF/datasets'
    dataset_dir = os.path.join(base_dir, 'EXIST 2025 Tweets Dataset')
    
    if args.all or (not args.input and not args.output):
        # 清洗训练集
        train_input = os.path.join(dataset_dir, 'training', 'EXIST2025_training.json')
        train_output = os.path.join(dataset_dir, 'training', 'EXIST2025_training_clean.json')
        if os.path.exists(train_input):
            clean_dataset(train_input, train_output)
        else:
            print(f"警告: 训练集文件不存在: {train_input}")
        
        # 清洗开发集
        dev_input = os.path.join(dataset_dir, 'dev', 'EXIST2025_dev.json')
        dev_output = os.path.join(dataset_dir, 'dev', 'EXIST2025_dev_clean.json')
        if os.path.exists(dev_input):
            clean_dataset(dev_input, dev_output)
        else:
            print(f"警告: 开发集文件不存在: {dev_input}")
                
        # 清洗测试集
        test_input = os.path.join(dataset_dir, 'test', 'EXIST2025_test_clean.json')
        test_output = os.path.join(dataset_dir, 'test', 'EXIST2025_test_clean_clean.json')
        if os.path.exists(test_input):
            clean_dataset(test_input, test_output)
        else:
            print(f"警告: 测试集文件不存在: {dev_input}")
     
    
    elif args.input and args.output:
        clean_dataset(args.input, args.output)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()