import os
import numpy as np
import torch
from src.models import TrainTranAD
import time, random
from data_config import *

class Config:
    x_dims = feature_dim
    z_dims = global_z_dim
    max_epochs = global_epochs
    batch_size = global_batch_size
    window_size = global_window_size
    exp_dir = exp_dir
    save_dir = exp_dir/ 'model'
    result_dir = exp_dir/'result'


def func_a(cluster, config: Config):
    total_train_time = 0
    model = TrainTranAD(
        feats=config.x_dims,
        max_epoch=config.max_epochs,
    )

    # train
    train_id_list = cluster['train'][:train_num]
    offline_data_list = get_offline_data(data_index_list=train_id_list)
    preprocess_offline_data_list = []
    for offline_data in offline_data_list:
        # x_train, _ = preprocess_meanstd(offline_data.T, offline_data.T) # cluster_Bank_v2.py
        x_train, _ = preprocess_meanstd(offline_data, offline_data) # cluster_Bank_v3.py
        preprocess_offline_data_list.append(x_train)

    save_dir = config.save_dir/ f'cluster_{cluster["label"]}'
    save_dir.mkdir(parents=True, exist_ok=True)
    train_start = time.time()
    
    # print("preprocess_offline_data_list:", preprocess_offline_data_list[0].shape)
    
    for i, arr in enumerate(preprocess_offline_data_list):
        print(f"Sequence {i} shape: {arr.shape}")

    model.fit(preprocess_offline_data_list, save_dir=save_dir, valid_portion=0.3)
    train_end = time.time()
    total_train_time += train_end - train_start
    return total_train_time


def torch_seed():
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():

    total_train_time = 0
    
    for cluster in clusters:
        torch_seed()
        train_time=func_a(cluster, config)
        print(f"{cluster['center' if 'center' in cluster else 'label']}--{train_time}s")
        total_train_time+=train_time
    print(f'============== offline training end ============= \n exp_key: {exp_key} \ntotal_train_time: {total_train_time}')



if __name__ == '__main__':
    
    config = Config()
    main()

