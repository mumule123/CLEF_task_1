import os
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 从环境变量获取配置，如果不存在则使用默认值或抛出警告
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "DeepSeek-R1-Int8")
API_URL = os.getenv("API_URL", "https://deepseek.fosu.edu.cn/api/chat/completions")

if not API_KEY:
    print("Warning: API_KEY not found in environment variables. Please check your .env file.")




# gpt
# API_KEY = "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"
# MODEL_NAME = "gpt-4.1-2025-04-14"
# # API_URL = "https://api.siliconflow.cn/v1/chat/completions"
