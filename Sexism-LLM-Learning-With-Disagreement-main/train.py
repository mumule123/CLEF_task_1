import dataset
import engine
import torch
import pandas as pd
import numpy as np
import random
import config
from tqdm import tqdm
import argparse
from utils import save_preds, eval_preds

from model import TransforomerModel
import warnings
warnings.filterwarnings('ignore')
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import logging
logging.set_verbosity_error()


def train(df_train, df_val, task, epochs, transformer, max_len, batch_size, lr, drop_out, df_results, training_data):
    print(f"\n{'='*50}")
    print(f"开始训练任务: {task}")
    print(f"使用模型: {transformer}")
    print(f"训练数据: {training_data}")
    print(f"训练样本数: {len(df_train)}, 验证样本数: {len(df_val)}")
    print(f"{'='*50}\n")

    train_dataset = dataset.TransformerDataset(
        text=df_train[config.COLUMN_TEXT].values,
        target=df_train[config.COLUMN_LABELS + task].values,
        max_len=max_len,
        transformer=transformer
    )

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        num_workers = config.TRAIN_WORKERS
    )
    
    val_dataset = dataset.TransformerDataset(
        text=df_val[config.COLUMN_TEXT].values,
        target=df_val[config.COLUMN_LABELS + task].values,
        max_len=max_len,
        transformer=transformer
    )

    val_data_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, 
        batch_size=batch_size, 
        num_workers=config.VAL_WORKERS
    )

    # 设备选择逻辑
    if not config.DEVICE or config.DEVICE == 'max':
        if torch.cuda.is_available():
            print(f"将使用GPU ID: {config.GPU_IDS}")
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            print("未找到GPU,使用CPU训练")
    else:
        device = torch.device(config.DEVICE)
        print(f"使用指定设备: {device}")

    model = TransforomerModel(transformer, drop_out, number_of_classes=config.UNITS[task]) 
    
    # 如果使用多GPU,则设置DataParallel
    if config.DEVICE == 'max' and torch.cuda.is_available():
        # 使用指定的GPU ID
        model = torch.nn.DataParallel(model, device_ids=config.GPU_IDS)
        print(f"使用{len(config.GPU_IDS)}个GPU进行并行训练")
    
    model.to(device)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.001,},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
    ]

    num_train_steps = int(len(df_train) / batch_size * epochs)
    print(f"总训练步数: {num_train_steps}")
    optimizer = AdamW(optimizer_parameters, lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    
    best_val_icm = 0
    # training and evaluation loop
    for epoch in range(1, epochs+1):
        print(f"\n{'='*20} Epoch {epoch}/{epochs} {'='*20}")
        
        print("训练阶段...")
        no_train, pred_train, _ , loss_train = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        save_preds(no_train, pred_train, df_train, task, training_data, 'training', epoch, transformer)
        try:
            icm_soft_train = eval_preds(task, training_data, 'training', epoch, transformer)
            if icm_soft_train is None:
                print("警告: 训练评估返回None，使用默认值0.0")
                icm_soft_train = 0.0
        except Exception as e:
            print(f"训练评估发生错误: {str(e)}")
            icm_soft_train = 0.0
        
        print("验证阶段...")
        no_val, pred_val, _ , loss_val = engine.eval_fn(val_data_loader, model, device)
        save_preds(no_val, pred_val, df_val, task, training_data, 'dev', epoch, transformer)
        try:
            icm_soft_val = eval_preds(task, training_data, 'dev', epoch, transformer)
            if icm_soft_val is None:
                print("警告: 验证评估返回None，使用默认值0.0")
                icm_soft_val = 0.0
        except Exception as e:
            print(f"验证评估发生错误: {str(e)}")
            icm_soft_val = 0.0

        df_new_results = pd.DataFrame({'task':task,
                            'epoch':epoch,
                            'transformer':transformer,
                            'max_len':max_len,
                            'batch_size':batch_size,
                            'lr':lr,
                            'dropout':drop_out,
                            'icm_soft_train': icm_soft_train,
                            'loss_train':loss_train,
                            'icm_soft_val':icm_soft_val,
                            'loss_val':loss_val
                        }, index=[0]
        )
        
        df_results = pd.concat([df_results, df_new_results], ignore_index=True)
        
        tqdm.write(f"\n性能指标:")
        tqdm.write(f"训练集 - ICM-soft = {icm_soft_train:.3f}, Loss = {loss_train:.3f}")
        tqdm.write(f"验证集 - ICM-soft = {icm_soft_val:.3f}, Loss = {loss_val:.3f}")

        # save models weights
        path_model_save = config.LOGS_PATH + '/model' + '_' + task + '_' + training_data + '_' + transformer.split("/")[-1] + '.pt'
        if training_data == 'training-dev' and  epoch == epochs:
            print(f"\n保存最终模型到: {path_model_save}")
            torch.save(model, path_model_save)
        else:
            if epoch == 1 or icm_soft_val > best_val_icm:
                best_val_icm = icm_soft_val
                print(f"\n发现最佳模型! 保存到: {path_model_save}")
                torch.save(model, path_model_save)

    print(f"\n训练完成! 最佳验证集ICM-soft: {best_val_icm:.3f}")
    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data', type=str, help='Datasets to train the models',default='training')
    args = parser.parse_args()
    
    if args.training_data == 'training':
        datasets = config.DATASET_TRAIN
        
    elif args.training_data == 'training-dev':
        datasets = config.DATASET_TRAIN_DEV
    
    elif not args.training_data:
        print('Specifying --training_data is required')
        exit(1)
    
    else:
        print('Specifying --training_data training OR training-dev')
        exit(1)

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)

    df_train = pd.read_csv(config.DATA_PATH + '/' + datasets, index_col=None).iloc[:config.N_ROWS]
    df_train['NO_value'] = df_train['soft_label_task1'].apply(lambda x: eval(x)['NO'])
    
    df_val = pd.read_csv(config.DATA_PATH + '/' + config.DATASET_DEV, index_col=None).iloc[:config.N_ROWS]
    df_val['NO_value'] = df_val['soft_label_task1'].apply(lambda x: eval(x)['NO']) 


    for transfomer in tqdm(config.TRANSFORMERS, desc='TRANSFORMERS', position=0):
        for task in tqdm(config.LABELS, desc='TASKS', position=1):
            
            if args.training_data == 'training-dev':
                df_info = pd.read_csv(config.LOGS_PATH + '/training_' + task + '_' + transfomer + '.csv')
                epochs = df_info.at[df_info['icm_soft_val'].idxmax(), 'epoch']
            else:
                epochs = config.EPOCHS
            
            df_results = pd.DataFrame(columns=['task',
                                    'epoch',
                                    'transformer',
                                    'max_len',
                                    'batch_size',
                                    'lr',
                                    'dropout',
                                    'icm_soft_train',
                                    'loss_train',
                                    'icm_soft_val',
                                    'loss_val'])
        
            tqdm.write(f'\nTask: {task} Data: {args.training_data} Transfomer: {transfomer.split("/")[-1]} Max_len: {config.MAX_LEN} Batch_size: {config.BATCH_SIZE} Dropout: {config.DROPOUT} lr: {config.LR}')
            
            df_results = train(df_train,
                                df_val,
                                task,
                                epochs,
                                transfomer,
                                config.MAX_LEN,
                                config.BATCH_SIZE,
                                config.LR,
                                config.DROPOUT,
                                df_results,
                                args.training_data
            )
                
            df_results.to_csv(config.LOGS_PATH + '/' + args.training_data + '_' + task + '_' + transfomer + '.csv', index=False)