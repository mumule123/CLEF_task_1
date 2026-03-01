"""
该脚本实现了基于PEFT库的LoRA微调功能，用于EXIST性别歧视检测任务
"""

# 导入必要的库
import os
import json
import pandas as pd
import torch
from api_key import MODEL_NAME
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType, 
    prepare_model_for_kbit_training
)
from util import json_df
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EXISTDataset(Dataset):
    """EXIST数据集的PyTorch数据集类"""
    
    def __init__(self, dataframe, tokenizer, max_length=512, prompt_template=None):
        """
        初始化数据集
        
        参数:
            dataframe (pd.DataFrame): 包含'tweet'和'task1_binary'列的数据框
            tokenizer: HuggingFace tokenizer
            max_length (int): 输入序列的最大长度
            prompt_template (str, optional): 提示模板，默认为None
        """
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template if prompt_template else "Tweet: {text}\nIs this sexist? Answer: {label}"
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        tweet = self.dataframe.iloc[idx]['tweet']
        label = self.dataframe.iloc[idx]['task1_bin']
        
        # 映射标签为文本格式
        label_text = "YES" if label == "sexist" else "NO"
        
        # 构建提示
        prompt = self.prompt_template.format(text=tweet, label=label_text)
        
        # 编码输入文本
        encodings = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze()  # 对于自回归LM，标签与输入相同
        }

def load_data(json_path):
    """加载EXIST数据集并进行预处理"""
    df = json_df(json_path)
    
    # 确保数据集具有必要的列
    required_cols = ['tweet', 'task1_bin']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"数据集缺少必要的列：{col}")
    
    return df

def create_prompt_template(system_prompt, response_format):
    """创建用于训练的提示模板"""
    template = f"""### System: {system_prompt}

### User: {{text}}

### Assistant: {{label}}"""
    return template

def setup_tokenizer(model_name_or_path):
    """配置tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # 确保EOS和BOS标记存在
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.pad_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    
    # 设置填充标记为EOS标记
    tokenizer.padding_side = 'right'
    
    return tokenizer

def setup_lora_model(model_name_or_path, lora_r=8, lora_alpha=16, lora_dropout=0.05):
    """配置LoRA模型"""
    # 检查CUDA可用性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
        device_map="auto" if device == "cuda" else None
    )
    
    # 为量化训练准备模型
    model = prepare_model_for_kbit_training(model)
    
    # 定义LoRA配置
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 通常的LoRA目标模块
    )
    
    # 应用LoRA配置到模型
    model = get_peft_model(model, peft_config)
    
    # 打印可训练的参数数量
    model.print_trainable_parameters()
    
    return model

def finetune_model(
    model_name_or_path,
    train_df,
    output_dir,
    system_prompt,
    response_format,
    epochs=1,
    batch_size=4,
    learning_rate=2e-4,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    max_length=512
):
    """使用LoRA微调模型"""
    # 设置tokenizer
    tokenizer = setup_tokenizer(model_name_or_path)
    
    # 创建提示模板
    prompt_template = create_prompt_template(system_prompt, response_format)
    
    # 设置数据集
    train_dataset = EXISTDataset(train_df, tokenizer, max_length, prompt_template)
    
    # 设置LoRA模型
    model = setup_lora_model(
        model_name_or_path, 
        lora_r=lora_r, 
        lora_alpha=lora_alpha, 
        lora_dropout=lora_dropout
    )
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 非掩码语言模型
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
        label_names=['labels']
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # 开始训练
    logger.info("开始LoRA微调...")
    trainer.train()
    
    # 保存LoRA适配器
    adapter_path = f"{output_dir}/lora_adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info(f"LoRA适配器已保存到 {adapter_path}")
    
    return model, tokenizer

def main(dataset_path, model_name_or_path, output_dir, system_prompt, response_format):
    """主函数"""
    # 加载数据
    df = load_data(dataset_path)
    logger.info(f"加载了 {len(df)} 条数据样本")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 微调模型
    model, tokenizer = finetune_model(
        model_name_or_path=model_name_or_path,
        train_df=df,
        output_dir=output_dir,
        system_prompt=system_prompt,
        response_format=response_format,
        epochs=1,
        batch_size=4,
        learning_rate=2e-4,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05
    )
    
    logger.info("LoRA微调完成")

if __name__ == "__main__":
    from prompts import PROMPT_VALUE, RESPONSE_VALUE
    
    # 获取当前脚本所在目录的路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # 数据集路径
    dataset_path = os.path.join(parent_dir, 'datasets', 'EXIST2025_dev_gold_all.json')
    
    # 模型名称
    model_name = MODEL_NAME 
    
    # 输出目录
    output_dir = os.path.join(parent_dir, 'lora_models')
    
    # 开始微调
    main(
        dataset_path=dataset_path,
        model_name_or_path=model_name,
        output_dir=output_dir,
        system_prompt=PROMPT_VALUE,
        response_format=RESPONSE_VALUE
    )
