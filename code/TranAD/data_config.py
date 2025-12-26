import pathlib
import json
import numpy as np
import os
import argparse
import torch
from sklearn.preprocessing import MinMaxScaler

# ======================
# 工具函数（提前定义）
# ======================

def preprocess_minmax(df_train, df_test):
    """
    使用 MinMaxScaler 对训练和测试数据进行归一化（范围 [-1, 1]）
    """
    print('minmax', end=' ')
    df_train = np.asarray(df_train, dtype=np.float32)
    df_test = np.asarray(df_test, dtype=np.float32)
    if len(df_train.shape) == 1 or len(df_test.shape) == 1:
        raise ValueError('Data must be a 2-D array')
    if np.any(np.isnan(df_train)):
        print('train data contains null values. Will be replaced with 0')
        df_train = np.nan_to_num(df_train)
    if np.any(np.isnan(df_test)):
        print('test data contains null values. Will be replaced with 0')
        df_test = np.nan_to_num(df_test)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(df_train)
    df_train = scaler.transform(df_train)
    df_test = scaler.transform(df_test)
    return df_train, df_test


def preprocess_meanstd(df_train, df_test):
    """
    使用均值-标准差标准化，并进行 3-sigma 截断
    """
    # print('meanstd', end=' ')
    df_train = np.asarray(df_train, dtype=np.float32)
    if len(df_train.shape) == 1:
        raise ValueError('Data must be a 2-D array')
    if np.any(np.isnan(df_train)):
        print('Data contains null values. Will be replaced with 0')
        df_train = np.nan_to_num(df_train)

    # k = 3
    k = 2
    e = 1e-3
    mean_array = np.mean(df_train, axis=0, keepdims=True)
    std_array = np.std(df_train, axis=0, keepdims=True)
    std_array[std_array == 0] = e

    # 3-sigma 截断
    df_train = np.clip(df_train, mean_array - k * std_array, mean_array + k * std_array)

    train_mean = np.mean(df_train, axis=0, keepdims=True)
    train_std = np.std(df_train, axis=0, keepdims=True)
    train_std[train_std == 0] = e

    df_train_new = (df_train - train_mean) / train_std

    df_test = np.clip(df_test, train_mean - k * train_std, train_mean + k * train_std)
    df_test_new = (df_test - train_mean) / train_std

    return df_train_new, df_test_new


# ======================
# 参数解析与路径设置
# ======================

project_path = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default="/root/shared-nvme/work/timeSeries/OmniTransfer_new/code/test_dataset/data1")
# GPU option
parser.add_argument('--gpu_id', type=int, default=1)
# dataset
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=250)
parser.add_argument('--base_model_dir', type=str, default=None)

# model
parser.add_argument('--z_dim', type=int, default=3)
parser.add_argument('--window_size', type=int, default=60)
parser.add_argument('--layer_num', type=int, default=1)
parser.add_argument('--tran_dim', type=int, default=16)

# training
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--start_epoch', type=int, default=1)
parser.add_argument('--valid_epoch', type=int, default=10)
parser.add_argument('--index_weight', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--train_type', type=str, required=True)
parser.add_argument('--freeze_index_list_encoder', type=int, nargs='+')
parser.add_argument('--freeze_index_list_decoder', type=int, nargs='+')
parser.add_argument('--training_period', type=int, default=None)

parser.add_argument('--min_std', type=float, default=0.01)
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--train_num', type=int, default=5)
parser.add_argument('--index_weight_index', type=int, default=1)
parser.add_argument('--use_center_score_path', type=str, default='')
parser.add_argument('--epsilon', type=float, default=0.95)

args = parser.parse_args()

# ======================
# 配置变量（仅 data1）
# ======================


train_num = args.train_num
min_std = args.min_std
dataset_path = args.dataset_path
index_weight_index = args.index_weight_index
use_center_score_path = args.use_center_score_path

single_score_th = 10000
out_dir = args.out_dir
GPU_index = str(args.gpu_id)
global_device = torch.device(f'cuda:{GPU_index}')
# global_device = 'cpu'  # 强制使用 CPU

global_epochs = args.epochs
global_start_epoch = args.start_epoch
seed = args.seed
training_period = args.training_period

global_z_dim = args.z_dim
global_batch_size = args.batch_size
global_learning_rate = args.lr
circle_loss_weight = args.index_weight
global_layer_num = args.layer_num
global_tran_dim = args.tran_dim
train_type = args.train_type
global_epsilon = args.epsilon

learning_rate_decay_factor = 1
global_valid_epoch_freq = args.valid_epoch
global_ws = args.window_size

# 构建实验 key 和目录
exp_key = train_type
exp_key += f"_{train_num}nodes"
exp_key += f"_{index_weight_index}iwi"
exp_key += f"_{min_std}clip"
exp_key += f"_{global_layer_num}l"
exp_key += f"_{global_tran_dim}dim"
exp_key += f"_{training_period}daytrain"
exp_key += f"_{global_learning_rate}lr"
exp_key += f"_{global_epochs}epoch"
exp_key += f"_{global_batch_size}bs"
exp_key += f"_{global_ws}ws"
exp_key += f"_{global_epsilon}eps"

exp_dir = pathlib.Path(out_dir) / "TranAD" / exp_key

if 'initr' in exp_key:
    global_learning_rate = 5e-3

base_model_dir = args.base_model_dir

dataset_root = pathlib.Path(args.dataset_root)
# dataset_root = pathlib.Path(f"code/test_dataset/data1")
print("dataset_root:", dataset_root)
online_data_path = dataset_root / "online_data.npy"
offline_data_path = dataset_root / "offline_data.npy"
cluster_json_path = dataset_root / 'cluster.json'

global_window_size = 13
online_data = np.load(online_data_path)
feature_dim = online_data.shape[-1]  # 或 online_data.shape[2]
bf_search_min = 0
bf_search_max = 400
bf_search_step_size = 1
noshare_save_dir = project_path / base_model_dir
eval_item_length = 360 * 2 - (global_window_size - 1)

# ======================
# data1 专用数据加载函数
# ======================

def get_data_by_index(data_index, training_period):
    data = np.load(online_data_path)
    test_data = data
    train_data = data[:, :, 1440 - 288 * training_period:1440]
    return train_data[data_index], test_data[data_index]


def get_offline_data(data_index_list):
    data = np.load(offline_data_path)
    return [data[index, :, :] for index in data_index_list]


def get_recon_index_weight():
    index_weight = np.ones(np.load(online_data_path).shape[2])
    return index_weight


index_loss_weight = get_recon_index_weight()

# ======================
# 加载 cluster 信息
# ======================

with open(cluster_json_path, 'r') as cl:
    cs = json.load(cl)

data_list = []
for c in cs:
    data_list.extend(c['test'])
eval_data_list = data_list

if 'noshare' not in exp_key:
    clusters = cs
else:
    clusters = [{"label": i, "center": i, "train": [i], "test": [i]} for i in data_list]
    
print("clusters:", clusters)
