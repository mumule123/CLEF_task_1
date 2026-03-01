import json
import os
from pathlib import Path

def count_languages(file_path):
    """
    统计JSON文件中的西班牙语和英语推文数量
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    spanish_count = 0
    english_count = 0
    
    for tweet_id, tweet_info in data.items():
        # 如果数据中不包含lang字段或为空，则跳过
        if 'lang' not in tweet_info or not tweet_info['lang']:
            continue
        
        if tweet_info['lang'] == 'es':
            spanish_count += 1
        elif tweet_info['lang'] == 'en':
            english_count += 1
    
    return spanish_count, english_count

def main():
    # 数据集所在目录
    dataset_dir = Path("/t-3058/CLEF/datasets/EXIST 2025 Tweets Dataset")
    
    # 训练集、开发集和测试集路径
    training_file = dataset_dir / "training" / "EXIST2025_training.json"
    dev_file = dataset_dir / "dev" / "EXIST2025_dev.json"
    test_file = dataset_dir / "test" / "EXIST2025_test_clean.json"
    
    # 初始化计数器
    total_spanish = 0
    total_english = 0
    
    # 处理训练集
    if training_file.exists():
        es_count, en_count = count_languages(training_file)
        total_spanish += es_count
        total_english += en_count
        print(f"训练集: 西班牙语: {es_count}条, 英语: {en_count}条, 总计: {es_count + en_count}条")
    else:
        print(f"训练集文件不存在: {training_file}")
    
    # 处理开发集
    if dev_file.exists():
        es_count, en_count = count_languages(dev_file)
        total_spanish += es_count
        total_english += en_count
        print(f"开发集: 西班牙语: {es_count}条, 英语: {en_count}条, 总计: {es_count + en_count}条")
    else:
        print(f"开发集文件不存在: {dev_file}")
    
    # 处理测试集
    if test_file.exists():
        es_count, en_count = count_languages(test_file)
        total_spanish += es_count
        total_english += en_count
        print(f"测试集: 西班牙语: {es_count}条, 英语: {en_count}条, 总计: {es_count + en_count}条")
    else:
        print(f"测试集文件不存在: {test_file}")
    
    # 输出总数
    print("\n总计:")
    print(f"西班牙语总数: {total_spanish}条")
    print(f"英语总数: {total_english}条")
    print(f"所有语言总数: {total_spanish + total_english}条")

if __name__ == "__main__":
    main()