import json
import os

def load_json_data(file_path):
    """加载JSON文件数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {e}")
        return None

def evaluate_predictions(gold_data_path, prediction_data_path):
    """
    评估预测结果与标准答案的准确率
    
    参数:
    gold_data_path (str): 标准答案JSON文件路径
    prediction_data_path (str): 预测结果JSON文件路径
    
    返回:
    dict: 包含评估结果的字典
    """
    # 加载数据
    gold_data = load_json_data(gold_data_path)
    pred_data = load_json_data(prediction_data_path)
    
    if not gold_data or not pred_data:
        return {"error": "无法加载数据文件"}
    
    # 创建ID到值的映射
    gold_map = {item["id"]: item["value"] for item in gold_data}
    
    # 对于预测结果，value是列表格式，需要取第一个元素
    pred_map = {item["id"]: item["value"] for item in pred_data}
    
    # 匹配数据
    total_matched = 0
    correct_predictions = 0
    matched_ids = []
    
    for id, gold_value in gold_map.items():
        if id in pred_map:
            total_matched += 1
            matched_ids.append(id)
            if pred_map[id] == gold_value:
                correct_predictions += 1
    
    # 计算准确率
    accuracy = correct_predictions / total_matched if total_matched > 0 else 0
    
    # 整合结果
    result = {
        "total_gold_records": len(gold_data),
        "total_prediction_records": len(pred_data),
        "matched_records": total_matched,
        "correct_predictions": correct_predictions,
        "incorrect_predictions": total_matched - correct_predictions,
        "accuracy": accuracy,
        "accuracy_percentage": f"{accuracy * 100:.2f}%"
    }
    
    return result

def main():
    # 使用os.path处理路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 文件路径
    gold_path = os.path.join(project_root, "datasets", "EXIST2025_dev_gold.json")
    pred_path = os.path.join(project_root, "model_out", "task1_hard_normalized_2.json")
    
    # 执行评估
    result = evaluate_predictions(gold_path, pred_path)
    
    # 打印结果
    print("\n评估结果:")
    print(f"标准答案记录总数: {result['total_gold_records']}")
    print(f"预测结果记录总数: {result['total_prediction_records']}")
    print(f"成功匹配的记录数: {result['matched_records']}")
    print(f"正确预测数: {result['correct_predictions']}")
    print(f"错误预测数: {result['incorrect_predictions']}")
    print(f"准确率: {result['accuracy_percentage']}")

if __name__ == "__main__":
    main()