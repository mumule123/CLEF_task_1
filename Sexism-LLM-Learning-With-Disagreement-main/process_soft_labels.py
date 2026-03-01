import json

def process_soft_labels(input_file, output_file):
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 创建输出格式的字典
    processed_data = {}
    
    # 处理每个条目
    for item in data:
        id_num = item['id']
        value = item['value']
        
        processed_data[id_num] = {
            "soft_label": {
                "YES": value.get("YES", 0.0),
                "NO": value.get("NO", 0.0)
            }
        }
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2)

if __name__ == "__main__":
    # 处理开发集
    process_soft_labels(
        'golds/EXIST2025_dev_task1_gold_soft.json', 
        'golds/EXIST2025_dev_task1_gold_soft_processed.json'
    )
    
    # 处理训练集
    process_soft_labels(
        'golds/EXIST2025_training_task1_gold_soft.json',
        'golds/EXIST2025_training_task1_gold_soft_processed.json'
    )