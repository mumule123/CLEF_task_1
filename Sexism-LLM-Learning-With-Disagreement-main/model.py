import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
import config
import os

class TransforomerModel(nn.Module):
    def __init__(self, transformer, drop_out, number_of_classes):
        super(TransforomerModel, self).__init__()
        self.number_of_classes = number_of_classes
        # 从本地模型路径加载配置
        model_path = os.path.join(config.MODEL_PATH, transformer)
        self.embedding_size = AutoConfig.from_pretrained(model_path).hidden_size
        # 从本地模型路径加载模型
        self.transformer = AutoModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(drop_out)
        self.classifier = nn.Linear(self.embedding_size * 2, self.number_of_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, iputs):
        transformer_output  = self.transformer(**iputs)
        mean_pool = torch.mean(transformer_output['last_hidden_state'], 1)
        max_pool, _ = torch.max(transformer_output['last_hidden_state'], 1)
        cat = torch.cat((mean_pool,max_pool), 1)
        drop = self.dropout(cat)
        output = self.classifier(drop)
        output = self.softmax(output)
        return output

