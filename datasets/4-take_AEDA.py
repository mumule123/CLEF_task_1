import os
import pandas as pd
import random
import re
import argparse
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
import string

# 下载nltk必需的资源
nltk.download('punkt')

# 定义可用于AEDA的标点符号
PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']

# 定义输入和输出文件路径
INPUT_FILES = {
    'training': os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                           "EXIST 2025 Tweets Dataset", "training", "EXIST2025_training_extracted.csv"),
    'dev': os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                       "EXIST 2025 Tweets Dataset", "dev", "EXIST2025_dev_extracted.csv"),
    'test': os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                        "EXIST 2025 Tweets Dataset", "test", "EXIST2025_test_extracted.csv"),
    'all': os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                       "EXIST2025_all_extracted.csv")
}

OUTPUT_FILES = {
    'training': os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                            "EXIST 2025 Tweets Dataset", "training", "EXIST2025_training_AEDA.csv"),
    'dev': os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                        "EXIST 2025 Tweets Dataset", "dev", "EXIST2025_dev_AEDA.csv"),
    'test': os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                         "EXIST 2025 Tweets Dataset", "test", "EXIST2025_test_AEDA.csv"),
    'all': os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                        "EXIST2025_all_AEDA.csv")
}

def clean_tweet(tweet):
    """
    清理推文文本，移除多余的引号和转义符
    """
    if not isinstance(tweet, str):
        return ""
    
    # 移除开头和结尾的额外引号
    tweet = re.sub(r'^"{2,}|"{2,}$', '', tweet)
    
    # 替换中间的重复引号为单个引号
    tweet = re.sub(r'"{2,}', '"', tweet)
    
    return tweet.strip()

def simple_sentence_split(text):
    """
    简单的句子分割函数，不依赖nltk的sent_tokenize
    
    参数:
        text (str): 需要分割的文本
        
    返回:
        list: 分割后的句子列表
    """
    # 使用常见的句子结束标记来分割文本
    sentence_endings = ['. ', '! ', '? ', '; ', '\n', '\t']
    
    # 如果文本很短或为空，直接返回整个文本作为一个句子
    if not text or len(text) < 50:
        return [text]
    
    # 初始化结果列表和当前句子
    sentences = []
    current_sentence = ""
    
    # 逐字符遍历文本
    for i in range(len(text)):
        current_sentence += text[i]
        
        # 检查是否遇到句子结束标记
        for ending in sentence_endings:
            end_len = len(ending)
            if i >= end_len - 1 and text[i-end_len+1:i+1] == ending:
                sentences.append(current_sentence)
                current_sentence = ""
                break
    
    # 添加最后一个句子（如果有）
    if current_sentence:
        sentences.append(current_sentence)
    
    # 如果没有找到任何句子分割点，则将整个文本作为一个句子
    if not sentences:
        sentences = [text]
    
    return sentences

def aeda_augmentation(text, p=0.3):
    """
    使用AEDA方法增强文本
    
    参数:
        text (str): 需要增强的文本
        p (float): 插入标点符号的概率，默认为0.3
        
    返回:
        str: 增强后的文本
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return text
    
    # 清理文本
    text = clean_tweet(text)
    
    # 使用自定义简单句子分割函数
    sentences = simple_sentence_split(text)
    augmented_sentences = []
    
    for sentence in sentences:
        # 使用空格分词（简单快速）
        tokens = sentence.split()
        if len(tokens) <= 1:
            augmented_sentences.append(sentence)
            continue
            
        # 计算要插入的标点符号数量
        num_insertions = max(1, int(len(tokens) * p))
        
        # 随机选择插入位置
        if len(tokens) > 1:
            insert_positions = random.sample(range(len(tokens)), min(num_insertions, len(tokens) - 1))
            insert_positions.sort()  # 排序位置以保持顺序
        else:
            insert_positions = []
            
        # 插入标点符号
        augmented_tokens = []
        for i, token in enumerate(tokens):
            if i in insert_positions:
                # 随机选择一个标点符号
                punct = random.choice(PUNCTUATIONS)
                augmented_tokens.append(punct)
            augmented_tokens.append(token)
            
        # 重新组合成句子
        augmented_sentence = ' '.join(augmented_tokens)
        # 修复标点符号周围的空格
        augmented_sentence = re.sub(r'\s+([,.!?;:])', r'\1', augmented_sentence)
        
        augmented_sentences.append(augmented_sentence)
    
    # 合并句子
    augmented_text = ' '.join(augmented_sentences)
    return augmented_text

def augment_dataframe(df):
    """
    对DataFrame中的推文进行AEDA增强
    
    参数:
        df (pandas.DataFrame): 包含推文的DataFrame
        
    返回:
        pandas.DataFrame: 增强后的DataFrame
    """
    # 创建新的DataFrame来存储增强后的数据
    augmented_df = df.copy()
    
    # 对'tweet'列进行AEDA增强
    tqdm.pandas(desc="应用AEDA增强")
    augmented_df['tweet_original'] = df['tweet'].apply(clean_tweet)
    augmented_df['tweet'] = df['tweet'].progress_apply(
        lambda x: aeda_augmentation(x) if isinstance(x, str) else x
    )
    
    return augmented_df

def process_file(input_file, output_file):
    """
    处理单个CSV文件，应用AEDA增强并保存结果
    
    参数:
        input_file (str): 输入CSV文件路径
        output_file (str): 输出CSV文件路径
    """
    print(f"处理文件: {input_file}")
    
    try:
        # 读取CSV文件
        df = pd.read_csv(input_file, sep='\t', quoting=1)
        
        # 应用AEDA增强
        augmented_df = augment_dataframe(df)
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存增强后的数据
        augmented_df.to_csv(output_file, sep='\t', index=False, quoting=1, encoding='utf-8')
        
        print(f"增强数据已保存到: {output_file}")
        
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用AEDA技术对EXIST 2025数据集进行数据增强')
    parser.add_argument('--dataset', type=str, choices=['training', 'dev', 'test', 'all'], 
                        default='all', help='要处理的数据集')
    parser.add_argument('--probability', type=float, default=0.3, 
                        help='插入标点符号的概率 (0.0-1.0)')
    
    args = parser.parse_args()
    
    print("开始AEDA数据增强...")
    print(f"选择的数据集: {args.dataset}")
    print(f"插入标点符号的概率: {args.probability}")
    
    if args.dataset == 'all':
        for dataset in ['training', 'dev', 'test', 'all']:
            if os.path.exists(INPUT_FILES[dataset]):
                process_file(INPUT_FILES[dataset], OUTPUT_FILES[dataset])
    else:
        # 处理指定的单个数据集
        if os.path.exists(INPUT_FILES[args.dataset]):
            process_file(INPUT_FILES[args.dataset], OUTPUT_FILES[args.dataset])
        else:
            print(f"错误: 找不到输入文件 {INPUT_FILES[args.dataset]}")
    
    print("AEDA数据增强完成!")

if __name__ == "__main__":
    main()