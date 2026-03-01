"""
该脚本实现了使用LoRA微调模型进行推理的功能
"""

# 导入必要的库
import os
import json
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import requests
from tqdm import tqdm
import time
import threading
import concurrent.futures
from api_key import API_KEY, MODEL_NAME, API_URL
from util import normalize_response, save_responses
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建线程本地存储对象
thread_local = threading.local()

def get_session():
    """为每个线程创建或获取一个Session对象"""
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
        thread_local.last_request_time = 0
        thread_local.request_count = 0
    return thread_local.session

def load_lora_model(model_name_or_path, adapter_path):
    """加载带有LoRA适配器的模型"""
    # 检查CUDA可用性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    # 加载LoRA适配器
    model = PeftModel.from_pretrained(model, adapter_path)
    
    logger.info(f"已加载带有LoRA适配器的模型：{model_name_or_path}")
    return model, tokenizer

def generate_response(model, tokenizer, tweet, system_prompt, max_new_tokens=100):
    """使用本地模型生成响应"""
    # 构建输入文本
    delimiter = "####"
    prompt = f"{system_prompt}\n\n{delimiter} {tweet} {delimiter}"
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 将输入移动到模型所在的设备
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 生成响应
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.1,  # 低温度使输出更加确定性
        )
    
    # 解码响应
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取用户输入之后的内容
    response = response.split(delimiter)[-1].strip()
    
    return response

def fetch_lora_response(tweet, df, idx, model, tokenizer, system_prompt, response_format):
    """使用本地LoRA微调模型获取性别歧视检测的响应"""
    try:
        # 获取提示
        prompt_value = system_prompt
        
        # 生成响应
        response = generate_response(model, tokenizer, tweet, f"{prompt_value} {response_format}")
        
        # 返回结果
        return {"tweet": tweet, "response": response, "index": idx}
    
    except Exception as e:
        logger.error(f"推文索引 {idx} 生成异常: {e}")
        return {"tweet": tweet, "response": f"错误: {str(e)}", "index": idx}

def main_lora(dataframe, model_name_or_path, adapter_path, system_prompt, response_format, max_workers=4):
    """使用本地LoRA微调模型并发处理推文"""
    tweets = dataframe['tweet'].to_list()
    raw_responses = {}   # 存储原始响应
    responses_dict = {}  # 存储规范化响应
    total_tweets = len(tweets)
    
    # 加载模型和tokenizer
    model, tokenizer = load_lora_model(model_name_or_path, adapter_path)
    
    # 创建线程池并发提交任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务给线程池
        future_to_index = {
            executor.submit(
                fetch_lora_response, 
                tweet, 
                dataframe, 
                i, 
                model, 
                tokenizer, 
                system_prompt, 
                response_format
            ): i 
            for i, tweet in enumerate(tweets)
        }

        # 使用tqdm显示进度
        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=total_tweets, desc="处理Task1-Hard模式(LoRA)"):
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
                logger.error(f"推文索引 {idx} 生成异常: {exc}")
                row_id = dataframe.iloc[idx]['id_EXIST']
                raw_responses[row_id] = {
                    'id': row_id,
                    'response': f"处理异常: {exc}",
                    'test_case': "EXIST2025",
                    'error': True
                }

    return responses_dict, raw_responses

def run_lora_inference():
    """运行LoRA推理流程的主函数"""
    # 获取当前脚本所在目录的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # 获取数据
    json_location = os.path.join(parent_dir, 'datasets', 'EXIST2025_dev_all.json')
    
    # 导入提示
    from prompts import PROMPT_VALUE, RESPONSE_VALUE
    
    # 加载数据
    from util import json_df
    df = json_df(json_location)
    
    # 模型路径
    model_name =   MODEL_NAME# 根据实际使用的模型调整
    adapter_path = os.path.join(parent_dir, 'lora_models', 'lora_adapter')
    
    # 设置输出目录
    output_dir = os.path.join(parent_dir, 'model_out')
    os.makedirs(output_dir, exist_ok=True)
    
    # 执行LoRA推理
    responses, raw_responses = main_lora(
        dataframe=df,
        model_name_or_path=model_name,
        adapter_path=adapter_path,
        system_prompt=PROMPT_VALUE,
        response_format=RESPONSE_VALUE,
        max_workers=4  # 根据GPU资源调整
    )
    
    # 保存原始响应
    if raw_responses:
        save_responses(
            raw_responses, 
            output_dir=output_dir, 
            base_filename=f"task1_hard_raw_lora", 
            increment_filename=True
        )
        
    # 保存规范化的响应
    if responses:
        save_responses(
            responses, 
            output_dir=output_dir, 
            base_filename=f"task1_hard_normalized_lora", 
            increment_filename=True
        )

if __name__ == "__main__":
    run_lora_inference()
