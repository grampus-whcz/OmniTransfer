import pathlib
import json
import numpy as np
import os
import argparse
import torch
from sklearn.preprocessing import MinMaxScaler

def get_data_by_index(data_index, training_period):
    data = np.load(online_data_path)
    test_data = data
    train_data = data[:, :, 1440-288*training_period:1440]
    return train_data[data_index], test_data[data_index]

def get_offline_data(data_index_list):
    data = np.load(offline_data_path)
    return [data[index, :, :] for index in data_index_list]

def get_recon_index_weight():
    # index_weight = np.ones(np.load(online_data_path).shape[1])  bug
    index_weight = np.ones(np.load(online_data_path).shape[2])
    print(index_weight)
    return index_weight

def preprocess_meanstd(df_train, df_test):
    # return preprocess_meanstd_item(df_train, df_test)
    return preprocess_minmax(df_train, df_test)

def preprocess_minmax(df_train, df_test):
    """
    normalize raw data
    """
    # print('minmax', end=' ')
    df_train = np.asarray(df_train, dtype=np.float32)
    df_test = np.asarray(df_test, dtype=np.float32)
    if len(df_train.shape) == 1 or len(df_test.shape) == 1:
        raise ValueError('Data must be a 2-D array')
    if np.any(sum(np.isnan(df_train)) != 0):
        print('train data contains null values. Will be replaced with 0')
        df_train = np.nan_to_num()
    if np.any(sum(np.isnan(df_test)) != 0):
        print('test data contains null values. Will be replaced with 0')
        df_test = np.nan_to_num()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(df_train)
    df_train = scaler.transform(df_train)
    df_test = scaler.transform(df_test)
    df_test = np.clip(df_test, a_min=-3.0, a_max=3.0)
    return df_train, df_test

def preprocess_meanstd_item(df_train, df_test):
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



project_path = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent
parser = argparse.ArgumentParser()
# GPU option
parser.add_argument('--gpu_id', type=int, default=0)
# dataset
parser.add_argument('--out_dir', type=str)
parser.add_argument('--base_model_dir', type=str)
parser.add_argument('--batch_size', type=int, default=64)

# model
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--z_dim', type=int, default=3)
parser.add_argument('--window_size', type=int, default=60)

# training
parser.add_argument('--epochs', type=int, default= 50)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=None)  # 409，2021，2022
parser.add_argument('--train_type', type=str)
parser.add_argument('--training_period', type=int, default=None) 
parser.add_argument('--valid_step', type=int, default=200) 
parser.add_argument('--valid_epoch', type=int, default=5) 
parser.add_argument('--dataset_path',type=str)
parser.add_argument('--train_num',type=int,default=5)
parser.add_argument('--index_weight_index',type=int,default=1)
args = parser.parse_args()

single_score_th = 10000
out_dir = args.out_dir

GPU_index = str(args.gpu_id)
global_device = torch.device(f'cuda:{GPU_index}')
# global_device = torch.device(f'cpu')

global_epochs = args.epochs
seed = args.seed
training_period = args.training_period

global_alpha = args.alpha
global_beta = args.beta
global_z_dim= args.z_dim
global_batch_size= args.batch_size
# learning rate
global_lr = args.lr

train_type = args.train_type
base_model_dir = args.base_model_dir
global_valid_step_freq = args.valid_step
global_valid_epoch_freq = args.valid_epoch
dataset_path = args.dataset_path
index_weight_index = args.index_weight_index
train_num = args.train_num
exp_key = train_type
exp_key += f"_{seed}"
exp_key += f"_{global_z_dim}z"
exp_key += f"_{train_num}nodes"
exp_key += f"_{index_weight_index}iwi"
exp_key +=f"_{training_period}daytrain"
exp_key +=f"_{global_lr}lr"
exp_key +=f"_{global_epochs}epoch"
exp_key +=f"_{args.window_size}ws"
exp_dir = pathlib.Path(out_dir) /"USAD" / exp_key

learning_rate_decay_by_step = 10000000
learning_rate_decay_factor = 1

dataset_root = pathlib.Path(f"code/test_dataset/data1")
online_data_path = dataset_root/"online_data.npy"
offline_data_path = dataset_root / "offline_data.npy"
cluster_json_path = dataset_root/'cluster.json'

index_loss_weight = get_recon_index_weight()
# print(f"index_loss_weight:{index_loss_weight}")


preprocess_days = 5
global_valid_step_freq = 500
global_window_size = 60
online_data = np.load(online_data_path)
feature_dim = online_data.shape[-1] 

cluster_json_path = cluster_json_path
bf_search_min = 0
bf_search_max = 400
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

