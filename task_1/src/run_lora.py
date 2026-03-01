"""
该脚本集成了LoRA微调和推理的完整流程
"""

import os
import argparse
import logging
from prompts import PROMPT_VALUE, RESPONSE_VALUE
from lora_finetune import main as finetune_main
from lora_inference import run_lora_inference

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LoRA微调和推理流程")
    parser.add_argument("--mode", type=str, choices=["finetune", "inference", "all"], default="all",
                        help="选择执行模式: finetune(仅微调), inference(仅推理), all(微调+推理)")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
                        help="要使用的基础模型名称或路径")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA缩放参数")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout比率")
    parser.add_argument("--batch_size", type=int, default=4, help="训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮次")
    return parser.parse_args()

def main():
    """主执行函数"""
    args = parse_args()
    
    # 获取当前脚本所在目录的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # 数据集路径
    train_dataset_path = os.path.join(parent_dir, 'datasets', 'EXIST2025_dev_gold_all.json')
    test_dataset_path = os.path.join(parent_dir, 'datasets', 'EXIST2025_dev_all.json')
    
    # 输出目录
    output_dir = os.path.join(parent_dir, 'lora_models')
    os.makedirs(output_dir, exist_ok=True)
    
    # 微调流程
    if args.mode in ["finetune", "all"]:
        logger.info("开始LoRA微调流程...")
        finetune_main(
            dataset_path=train_dataset_path,
            model_name_or_path=args.model_name,
            output_dir=output_dir,
            system_prompt=PROMPT_VALUE,
            response_format=RESPONSE_VALUE,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        logger.info("LoRA微调完成")
    
    # 推理流程
    if args.mode in ["inference", "all"]:
        logger.info("开始LoRA推理流程...")
        run_lora_inference()
        logger.info("LoRA推理完成")
    
    logger.info("全部流程执行完毕")

if __name__ == "__main__":
    main()
