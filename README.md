# EXIST 2025 Shared Task - Sexism Identification (Task 1)

This project contains the implementation for the EXIST 2025 Shared Task 1 on Sexism Identification. It supports both API-based classification (e.g., DeepSeek) and local inference using LoRA fine-tuned models.

## Project Structure

```
├── datasets/               # Dataset files (JSON)
│   ├── EXIST2025_dev_all.json
│   └── EXIST2025_dev_gold_all.json
├── evaluate/               # Evaluation scripts
│   ├── test_accuracy.py
│   └── test_icm.py
├── model_out/              # Directory for model outputs
├── src/                    # Source code
│   ├── api_key.py          # API configuration (loads from .env)
│   ├── lora_finetune.py    # LoRA fine-tuning script
│   ├── lora_inference.py   # LoRA inference script
│   ├── main.py             # Main entry point for API inference & LoRA usage
│   ├── prompts.py          # Prompts for LLMs
│   ├── run_lora.py         # Orchestrator for LoRA training/inference
│   └── util.py             # Utility functions
├── .env.example            # Template for environment variables
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Setup

1.  **Install Dependencies**

    Ensure you have Python 3.8+ installed.

    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**

    Copy `.env.example` to `.env` and fill in your API keys and configuration.

    ```bash
    cp .env.example .env
    ```

    Edit `.env` with your actual keys:

    ```ini
    API_KEY=your_api_key_here
    MODEL_NAME=DeepSeek-R1-Int8
    API_URL=https://deepseek.fosu.edu.cn/api/chat/completions
    ```

## Usage

### 1. API-based Inference (Task 1 - Hard)

To run classification using the configured API (e.g., DeepSeek):

```bash
python src/main.py
```

This will:
-   Load the dataset from `datasets/EXIST2025_dev_all.json`.
-   Send requests to the API using prompts defined in `src/prompts.py`.
-   Save results to `model_out/`.

### 2. LoRA Fine-tuning

To fine-tune a model using LoRA adapters:

```bash
python src/run_lora.py --mode finetune --model_name "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
```

**Arguments:**
-   `--mode`: `finetune`, `inference`, or `all` (default: `all`).
-   `--model_name`: Base model to fine-tune.
-   `--lora_r`, `--lora_alpha`, `--lora_dropout`: LoRA parameters.
-   `--batch_size`, `--learning_rate`, `--epochs`: Training hyperparameters.

### 3. LoRA Inference

To run inference using a fine-tuned LoRA model:

**Using `run_lora.py`:**

```bash
python src/run_lora.py --mode inference
```

**Using `main.py` (Integration):**

```bash
python src/main.py --use_lora --lora_path /path/to/lora_adapter
```

If `--lora_path` is not specified, it defaults to `lora_models/lora_adapter`.

## Evaluation

Scripts for evaluation are located in the `evaluate/` directory.

```bash
# Example usage (adjust paths as needed)
python evaluate/test_accuracy.py
```

## License

[License Information Here]
