"""
该脚本处理EXIST数据集中的文本，执行性别歧视检测任务(Task1)
使用Hard评估模式，结果为二分类: [YES] 或 [NO]
支持使用API调用或本地LoRA微调模型进行推理
"""

# 导入必要的库
import pandas as pd
import json
import re
import time
import os
import argparse
from api_key import API_KEY, MODEL_NAME, API_URL
from util import normalize_response,save_responses,json_df
from prompts import PROMPT_VALUE, RESPONSE_VALUE, get_column_value
import concurrent.futures
import requests
from tqdm import tqdm
import threading
import random


# 创建线程本地存储对象，确保每个线程拥有自己的请求计数和时间戳
thread_local = threading.local()

# 为每个线程创建session
def get_session():
    """
    为每个线程创建或获取一个Session对象。
    这样可以确保同一线程内所有请求复用连接池。
    
    返回:
        requests.Session: 当前线程的专用Session对象
    """
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
        thread_local.last_request_time = 0  # 添加时间戳跟踪
        thread_local.request_count = 0      # 添加请求计数
    return thread_local.session




# 同步函数 - 获取性别歧视检测的API响应
def fetch_response(tweet, api_key, df, idx, max_retries=2):
    """
    根据给定参数从API获取性别歧视检测的响应 (Task1-Hard模式)。

    参数:
        tweet (str): 需要检测的推文文本。
        api_key (str): 用于访问API的API密钥。
        df (pandas.DataFrame): 包含学习水平和性别信息的DataFrame。
        idx (int): DataFrame中当前行的索引。
        max_retries (int): 最大重试次数。

    返回:
        dict: 包含推文和API响应的字典。

    异常:
        Exception: 如果在API请求期间发生错误。
    """
    api_key = API_KEY
    delimiter = "####"
    
    # 获取当前线程的session
    session = get_session()

    
    # 获取系统提示和响应格式
    prompt_value = PROMPT_VALUE
    
    column_value = get_column_value(df.study_levels_annotators.iloc[idx], df.gender_annotators.iloc[idx],df.ethnicities_annotators.iloc[idx]
                                    ,df.countries_annotators.iloc[idx],df.age_annotators.iloc[idx]
                                    )

    # Task1-Hard模式的响应格式要求
    response_value = RESPONSE_VALUE
    
    # 准备API请求
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": f"""{prompt_value} {column_value} {response_value}"""},
            # {"role": "system", "content": f"""{prompt_value} {response_value}"""},
            {"role": "user", "content": delimiter + " " + tweet + delimiter + " "}
        ]
    }
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # 使用带重试逻辑的请求
    retries = 0
    while retries < max_retries:
        try:
            # 使用线程本地的session发送请求
            response = session.post(API_URL, json=payload, headers=headers, timeout=60)
            
            # 处理常见错误
            if response.status_code == 429:  # 速率限制
                retry_after = float(response.headers.get('Retry-After', 1))
                print(f"速率限制触发，推文索引 {idx}，{retry_after} 秒后重试...")
                time.sleep(retry_after)
                retries += 1
                continue
                
            if response.status_code != 200:  # 其他HTTP错误
                error_message = response.text
                print(f"HTTP错误 {response.status_code} for tweet index {idx}: {error_message}")
                return {"tweet": tweet, "response": f"HTTP错误 {response.status_code}: {error_message}", "index": idx}
                
            # 解析成功响应
            result = response.json()
            if 'choices' in result and result['choices']:
                return {"tweet": tweet, "response": result['choices'][0]['message']['content'], "index": idx}
            else:
                print(f"意外的响应结构，推文索引 {idx}:", json.dumps(result, indent=4))
                return {"tweet": tweet, "response": "错误: 意外的API响应", "index": idx}
                
        except requests.exceptions.RequestException as e:
            retries += 1
            if retries >= max_retries:
                return {"tweet": tweet, "response": f"错误: {str(e)}", "index": idx}
            time.sleep(2 ** retries)  # 指数退避
            
        except Exception as e:
            return {"tweet": tweet, "response": f"错误: {str(e)}", "index": idx}
            
    print(f"达到最大重试次数，推文索引 {idx}.")
    return {"tweet": tweet, "response": "错误: 达到最大重试次数", "index": idx}


def main(dataframe, api_key, max_workers):
    """
    使用线程池并发获取和处理推文。

    参数:
        dataframe (pandas.DataFrame): 包含推文的数据框。
        api_key (str): 用于访问API的API密钥。
        max_workers (int): 线程池的最大工作线程数。

    返回:
        包含两个字典的元组:
        - responses_dict: 一个将行ID映射到归一化响应值的字典。
        - raw_responses: 一个将行ID映射到原始响应值的字典。
    """
    tweets = dataframe['tweet'].to_list()
    raw_responses = {}   # 存储原始响应
    responses_dict = {}  # 存储规范化响应
    total_tweets = len(tweets)
    
    
    # 创建线程池并发提交任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务给线程池
        future_to_index = {
            executor.submit(fetch_response, tweet, api_key, dataframe, i): i 
            for i, tweet in enumerate(tweets)
        }

        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=total_tweets, desc="处理Task1-Hard模式"):
            idx = future_to_index[future]
            
            try:
                response_data = future.result()
                
                # 处理有效响应
                if 'response' in response_data and not response_data['response'].startswith("错误:"):
                    # 规范化响应
                    normalized_values = normalize_response(response_data['response'])
                    
                    # 获取正确的id_EXIST
                    row_id = dataframe.iloc[idx]['id_EXIST']
                     # 存储原始响应
                    raw_responses[row_id] = {
                        'test_case': "EXIST2025",
                        'id': row_id,
                        'response': response_data['response'],
                    }
                    # 存储规范化响应
                    responses_dict[row_id] = {
                        'test_case': "EXIST2025",
                        'id': row_id,
                        'value': normalized_values
                    }
                    

                else:
                    row_id = dataframe.iloc[idx]['id_EXIST']
                    raw_responses[row_id] = {
                        'id': row_id,
                        'response': response_data.get('response', 'Future failed without specific error'),
                        'test_case': "EXIST2025",
                        'error': True
                    }

            except Exception as exc:
                # 处理异常情况
                print(f"推文索引 {idx} 生成异常: {exc}")
                row_id = dataframe.iloc[idx]['id_EXIST']
                raw_responses[row_id] = {
                    'id': row_id,
                    'response': f"处理异常: {exc}",
                    'test_case': "EXIST2025",
                    'error': True
                }

    return responses_dict, raw_responses


# 主执行函数
def run_exist_processing(use_lora=False, lora_model_path=None):
    """
    运行EXIST处理流程的主函数：
    1. 加载测试数据
    2. 调用API或LoRA模型进行性别歧视检测(Task1-Hard模式)
    3. 保存规范化和原始响应结果
    
    参数:
        use_lora (bool): 是否使用LoRA微调模型而非API
        lora_model_path (str): LoRA适配器路径，当use_lora=True时使用
    """
    # 获取当前脚本所在目录的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    current_dir = os.path.dirname(current_dir)
    # 获取数据
    json_location = os.path.join(current_dir, 'datasets','EXIST2025_dev_all.json')
    # 加载数据
    df = json_df(json_location)
    
    # 使用API密钥
    api_key = API_KEY
    
    # 设置输出目录 - 使用相对路径
    output_dir = os.path.join(current_dir,'model_out')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 执行主处理逻辑
    if use_lora:
        # 使用LoRA模型推理
        from lora_inference import main_lora
        
        # 如果未提供模型路径，使用默认路径
        if lora_model_path is None:
            lora_model_path = os.path.join(current_dir, 'lora_models', 'lora_adapter')
        
        # 默认模型名称
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"  # 根据实际使用的模型调整
        
        # 执行LoRA推理
        responses, raw_responses = main_lora(
            dataframe=df,
            model_name_or_path=model_name,
            adapter_path=lora_model_path,
            system_prompt=PROMPT_VALUE,
            response_format=RESPONSE_VALUE,
            max_workers=4  # 根据GPU资源调整
        )
    else:
        # 使用API推理
        responses, raw_responses = main(
            dataframe=df,
            api_key=api_key,
            max_workers=50,  #gpt3
        )
    
    # 保存原始响应
    if raw_responses:
        file_suffix = "lora" if use_lora else ""
        save_responses(
            raw_responses, 
            output_dir=output_dir, 
            base_filename=f"task1_hard_raw{file_suffix}", 
            increment_filename=True
        )
        
    # 保存规范化的响应
    if responses:
        file_suffix = "lora" if use_lora else ""
        save_responses(
            responses, 
            output_dir=output_dir, 
            base_filename=f"task1_hard_normalized{file_suffix}", 
            increment_filename=True
        )


# 程序入口点
if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="EXIST性别歧视检测任务")
    parser.add_argument("--use_lora", action="store_true", help="使用LoRA微调模型代替API调用")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA适配器路径")
    args = parser.parse_args()
    
    # 执行处理流程
    run_exist_processing(use_lora=args.use_lora, lora_model_path=args.lora_path)