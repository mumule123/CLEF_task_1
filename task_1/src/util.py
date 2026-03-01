import json
import os
import re
import pandas as pd

def normalize_response(response):
    """
    规范化API响应，提取最后方括号内的YES或NO结果。

    参数:
        response (str): 要规范化的响应。

    返回:
        str: 包含规范化响应的字符串('YES'或'NO'或'UNKNOWN')。
    """
    # 查找最后出现的方括号
    last_bracket_start = response.rfind('[')
    last_bracket_end = response.rfind(']')

    # 提取方括号中的内容
    if last_bracket_start != -1 and last_bracket_end != -1 and last_bracket_start < last_bracket_end:
        final_answer = response[last_bracket_start + 1:last_bracket_end].strip().upper()
        
        # 验证答案是否有效(YES或NO)
        if final_answer in ['YES', 'NO']:
            return final_answer
        else:
            print(f"⚠️ 最终答案不符合预期格式(YES/NO): '{final_answer}'")
            return 'UNKNOWN'
    else:
        # 找不到方括号
        print(f"⚠️ 未在响应中找到最后的方括号: {response}")
        return 'UNKNOWN'
    
def save_responses(responses_dict, output_dir, base_filename, increment_filename):
    """
    将响应保存到JSON文件。

    参数:
        responses_dict (dict): 包含要保存的响应的字典。
        output_dir (str, 可选): 保存JSON文件的目录。默认为'.'。
        base_filename (str, 可选): JSON文件的基本文件名。默认为'NO_FILE_NAME_RMIT'。
        increment_filename (bool, 可选): 如果文件已存在，是否增加文件名。默认为True。

    异常:
        ValueError: 如果responses_dict不是字典。
    """
    if not isinstance(responses_dict, dict):
        raise ValueError("responses_dict必须是一个字典。")

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{base_filename}.json" if not increment_filename else get_next_filename(output_dir, base_filename)
    full_path = os.path.join(output_dir, filename)

    # 将字典转换为列表以进行JSON转储
    responses_list = list(responses_dict.values())

    try:
        with open(full_path, 'w') as file:
            json.dump(responses_list, file, indent=4)
        print(f"数据已保存到 {filename}")
    except IOError as e:
        print(f"保存数据失败: {e}")
        
        
        
def json_df(json_location):
    with open(json_location, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame.from_dict(data, orient='index')
    return df



def get_next_filename(output_dir, base_filename):
    """
    获取指定输出目录中给定基本文件名的下一个可用文件名。

    参数:
        output_dir (str): 存储文件的目录。
        base_filename (str): 要使用的基本文件名。

    返回:
        str: 格式为"{base_filename}_{number}.json"的下一个可用文件名。
    """
    pattern = re.compile(rf"{re.escape(base_filename)}_(\d+).json")
    max_number = 0
    for filename in os.listdir(output_dir):
        match = pattern.match(filename)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number
    return f"{base_filename}_{max_number + 1}.json"