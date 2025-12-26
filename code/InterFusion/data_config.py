import math
import pathlib
import json
import numpy as np
import os
import argparse
import torch
from sklearn.preprocessing import MinMaxScaler

def preprocess_meanstd(df_train, df_test):
    """returns normalized and standardized data.
    """
    # print('meanstd', end=' ')
    df_train = np.asarray(df_train, dtype=np.float32)

    if len(df_train.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df_train)) != 0):
        print('Data contains null values. Will be replaced with 0')
        df_train = np.nan_to_num(df_train)

    k = 5
    e = 1e-3
    mean_array = np.mean(df_train, axis=0, keepdims=True)
    std_array = np.std(df_train, axis=0, keepdims=True)
    std_array[np.where(std_array==0)] = e
    df_train = np.where(df_train > mean_array + k * std_array, mean_array + k * std_array, df_train)
    df_train = np.where(df_train < mean_array - k * std_array, mean_array - k * std_array, df_train)
    
    train_mean_array = np.mean(df_train, axis=0, keepdims=True)
    train_std_array = np.std(df_train, axis=0, keepdims=True)
    train_std_array[np.where(train_std_array==0)] = e
    
    df_train_new = (df_train - train_mean_array) / train_std_array
    
    df_test = np.where(df_test > train_mean_array + k * train_std_array, train_mean_array + k * train_std_array, df_test)
    df_test = np.where(df_test < train_mean_array - k * train_std_array, train_mean_array - k * train_std_array, df_test)
    df_test_new = (df_test - train_mean_array) / train_std_array

    return df_train_new, df_test_new

def preprocess_minmax(df_train, df_test):
    scaler = MinMaxScaler().fit(df_train)
    train = scaler.transform(df_train)
    test = scaler.transform(df_test)
    test = np.clip(test, a_min=-3.0, a_max=3.0)
    return train, test

def get_data_by_index(data_index, training_period):
    data = np.load(online_data_path)
    test_data = data
    train_data = data[:, :, 1440-288*training_period:1440]
    return train_data[data_index], test_data[data_index]

def get_offline_data(data_index_list):
    data = np.load(offline_data_path)
    return [data[index, :, :] for index in data_index_list]


def get_recon_index_weight():
    # index_weight = np.ones(np.load(online_data_path).shape[1]) # bug 
    index_weight = np.ones(np.load(online_data_path).shape[2])  
    print(index_weight)
    return index_weight



project_path = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent
parser = argparse.ArgumentParser()
# GPU option
parser.add_argument('--dataset_root', type=str, default="/root/shared-nvme/work/timeSeries/OmniTransfer_new/code/test_dataset/data1")
parser.add_argument('--gpu_id', type=int, default=0)
# dataset
parser.add_argument('--out_dir', type=str)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--base_model_dir', type=str, default=None)

# model
parser.add_argument('--z1_dim', type=int, default=3)
parser.add_argument('--model_dim', type=int, default=100)

# training
parser.add_argument('--epochs', type=int, default= 60)
parser.add_argument('--pretrain_epochs', type=int, default= 60)
parser.add_argument('--train_lr', type=float, default= 5e-4)
parser.add_argument('--pretrain_lr', type=float, default= 8e-3)
parser.add_argument('--seed', type=int, default=None)  # 409，2021，2022
parser.add_argument('--train_type', type=str)
parser.add_argument('--training_period', type=int, default=None) 
parser.add_argument('--valid_epoch', type=int, default=5) 

parser.add_argument('--min_std', type=float, default=0) 
parser.add_argument('--dataset_path',type=str)
parser.add_argument('--train_num',type=int,default=5)
parser.add_argument('--index_weight_index',type=int,default=1)
parser.add_argument('--use_cluster_json_path',type=str,default='')

args = parser.parse_args()
use_cluster_json_path = args.use_cluster_json_path
train_num = args.train_num
min_std = args.min_std
dataset_path = args.dataset_path
index_weight_index = args.index_weight_index

single_score_th = 10000
out_dir = args.out_dir

GPU_index = str(args.gpu_id)
global_device = torch.device(f'cuda:{GPU_index}')
# global_device = 'cpu'  # 强制使用 CPU

global_train_epochs = args.epochs
global_pretrain_epochs = args.pretrain_epochs
seed = args.seed
training_period = args.training_period
global_z1_dim= args.z1_dim
global_rnn_dims= args.model_dim
global_dense_dims= args.model_dim
global_batch_size= args.batch_size
# learning rate
pretrain_lr = args.pretrain_lr
train_lr = args.train_lr
clip_std_min = math.exp(-5)
clip_std_max = math.exp(2)
if_freeze_seq = True
train_type = args.train_type

exp_key = train_type
exp_key += f"_{train_num}nodes"
exp_key += f"_{index_weight_index}iwi"
exp_key += f"_{min_std}clip"
exp_key += f"_{seed}"
exp_key +=f"_{training_period}daytrain"
exp_key += f"_modeldim{global_dense_dims}"
exp_key += f"_epoch{global_train_epochs}"
exp_key += f"_lr{train_lr}"
exp_dir = pathlib.Path(out_dir) /"InterFusion" / exp_key

base_model_dir = args.base_model_dir


learning_rate_decay_by_epoch = 10
learning_rate_decay_factor = 1
global_valid_epoch_freq = args.valid_epoch

dataset_root = pathlib.Path(args.dataset_root)
online_data_path = dataset_root/"online_data.npy"
offline_data_path = dataset_root / "offline_data.npy"
cluster_json_path = dataset_root/'cluster.json'

index_loss_weight = get_recon_index_weight()

online_data = np.load(online_data_path)
feature_dim = online_data.shape[-1] 
output_padding_list = [0, 1, 1]
global_z2_dim = 8
global_window_size = 60
cluster_json_path = cluster_json_path
bf_search_min = 0
bf_search_max = 1000
bf_search_step_size = 1
noshare_save_dir = project_path / base_model_dir
eval_item_length = 288*2 - (global_window_size - 1)

data_list = []
with open(cluster_json_path, 'r') as cl:
    cs = json.load(cl)
for c in cs:
    data_list.extend(c['test'])  
eval_data_list = data_list
if 'noshare' not in exp_key:
    clusters = cs
else:
    clusters = [{"label": i, "center": i, "train": [i], "test":[i]} for i in data_list]




