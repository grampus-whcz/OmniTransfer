#!/usr/bin/env python
# coding: utf-8

import sys
import os
import pathlib
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter
from joblib import Parallel, delayed
from tqdm import tqdm
import argparse
import sklearn.cluster as sc

# ----------------------------
# 项目路径设置与模块导入
# ----------------------------
project_path = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent
sys.path.append(project_path / "code/cluster")

import omnicluster.feature_selection as fs
import omnicluster.preprocess as pre
import omnicluster.yin as yin
import omnicluster.train_1dcnn_ae as ae


# ----------------------------
# 工具函数定义（保持不变）
# ----------------------------

def moving_average(in_data, window=12, min_periods=1):
    """
    对输入数据进行滑动平均平滑处理。
    :param in_data: 输入的三维数组 (num_entity, num_features, time_steps)
    :param window: 滑动窗口大小
    :param min_periods: 最小观测数
    :return: 平滑后的数据（原地修改）
    """
    for data in in_data:
        for i, n_index_data in enumerate(data):
            df = pd.Series(n_index_data)
            moving_avg = df.rolling(window=window, min_periods=min_periods).mean()
            data[i] = moving_avg
    return in_data


def euc(x, y, x_index, y_index, weight_item):
    """
    带权重的欧氏距离计算。
    :param x, y: 两个样本，形状为 (num_features, time_steps)
    :param weight_item: 时间步权重向量
    :return: 加权欧氏距离标量
    """
    return np.sum(np.linalg.norm(x - y, axis=1) * weight_item)


def cal_distance(data, weight_item, distance_strategy='euc', save_path=None, prefix=""):
    """
    计算所有样本对之间的距离矩阵。
    :param data: 样本数据，形状为 (N, num_features, time_steps)
    :param weight_item: 时间步权重
    :param distance_strategy: 距离策略（目前仅支持 'euc'）
    :param save_path: 保存路径
    :param prefix: 文件名前缀
    """
    instance_pair_list = []
    n = data.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            instance_pair_list.append((i, j))

    distance_list = []
    for i, j in tqdm(instance_pair_list):
        if distance_strategy == 'euc':
            distance_list.append(euc(data[i], data[j], i, j, weight_item))

    distance_matrix = np.zeros((n, n))
    for idx, dis in enumerate(distance_list):
        i, j = instance_pair_list[idx]
        distance_matrix[i][j] = dis
        distance_matrix[j][i] = dis

    np.save(save_path / f'{distance_strategy}{prefix}.npy', distance_matrix)


def ncc(x, y, weight_item):
    """
    计算归一化互相关（Normalized Cross-Correlation）并返回最优相位偏移。
    :param x, y: 两个样本，形状为 (num_features, time_steps)
    :param weight_item: 时间步权重
    :return: 最优偏移量 s_max
    """
    norm_cc_list = []
    _, m = x.shape
    x_norm = np.linalg.norm(x, axis=1)
    for s in range(-m + 1, m):
        if s >= 0:
            y_s = np.hstack((y[:, m - s:], y[:, :m - s]))
        else:
            y_s = np.hstack((y[:, -s:], y[:, :-s]))
        corr = np.sum(np.multiply(x, y_s), axis=1) / (x_norm * np.linalg.norm(y_s, axis=1) + 1e-9)
        norm_cc_list.append(np.sum(corr * weight_item))
    max_index = np.argmax(norm_cc_list)
    s_max = list(range(-m + 1, m))[max_index]
    return s_max


def func_a(data_index, pvt, data, weight_item):
    """
    并行计算每个样本相对于枢纽点的最优相位偏移。
    """
    return ncc(data[pvt], data[data_index], weight_item)


def align_phase(euc_mat, data, weight_item, save_path, prefix="_pvt", n_jobs=30):
    """
    基于距离矩阵选择枢纽点（pvt），并对所有样本进行相位对齐。
    :param euc_mat: 距离矩阵
    :param data: 原始样本数据
    :param weight_item: 时间步权重
    :param save_path: 保存路径
    :param prefix: 文件名前缀
    :param n_jobs: 并行任务数
    :return: 偏移列表 s_list 和对齐后的数据
    """
    dis_sum = np.sum(euc_mat, axis=0)
    pvt = np.argmin(dis_sum)
    print(f"Chosen pivot index: {pvt}, distance sum: {dis_sum[pvt]}")

    s_list = Parallel(n_jobs=n_jobs)(
        delayed(func_a)(data_index, pvt, data, weight_item)
        for data_index in tqdm(range(data.shape[0]))
    )
    s_list = np.array(s_list)

    for data_index, s_max in enumerate(s_list):
        if s_max == 0:
            continue
        data[data_index] = np.concatenate([data[data_index, :, -(s_max):], data[data_index, :, 0:-(s_max)]], axis=-1)

    np.save(save_path / f'z_to_cluster_all{prefix}.npy', data)
    np.save(save_path / f's_list.npy', s_list)
    return s_list, data


def align_phase_online(data_online, data_offline, pvt, weight_item, save_path):
    """
    对在线数据进行相位对齐，参考离线数据的枢纽点。
    :param data_online: 在线数据
    :param data_offline: 离线对齐后数据
    :param pvt: 枢纽点索引
    :param weight_item: 时间步权重
    :param save_path: 保存路径
    :return: 偏移列表和对齐后的在线数据
    """
    s_list = np.zeros(data_online.shape[0])
    for data_index in range(data_online.shape[0]):
        s_max = ncc(data_offline[pvt], data_online[data_index], weight_item)
        s_list[data_index] = s_max
        if s_max != 0:
            data_online[data_index] = np.concatenate(
                [data_online[data_index, :, -(s_max):], data_online[data_index, :, 0:-(s_max)]], axis=-1
            )
    np.save(save_path / 'z_to_cluster_all_pvt_online.npy', data_online)
    np.save(save_path / 's_list_online.npy', s_list)
    return s_list, data_online


def cluster_main(distance, cluster_num, save_path, distance_strategy, prefix, outlier_threshold=10):
    """
    执行层次聚类，并将小簇标记为异常（-1），然后重新编号。
    :param distance: 距离矩阵
    :param cluster_num: 聚类数量
    :param save_path: 保存路径
    :param distance_strategy: 距离策略
    :param prefix: 文件名前缀
    :param outlier_threshold: 小于该数量的簇视为异常
    """
    model = sc.AgglomerativeClustering(
        n_clusters=cluster_num,
        compute_full_tree=True,
        affinity="precomputed",
        linkage="average"
    )
    pred = model.fit_predict(distance)
    pred_counter = Counter(pred)

    # 标记小簇为异常
    for label, count in pred_counter.items():
        if count <= outlier_threshold:
            pred[np.where(pred == label)] = -1

    # 重新编号（从1开始）
    new_pred_counter = Counter(pred)
    sorted_labels = sorted(new_pred_counter.keys())
    label_map = {old: new + 1 for new, old in enumerate(sorted_labels)}
    new_pred = np.array([label_map[x] for x in pred])

    print(f"Cluster result: {Counter(new_pred)}")
    np.save(
        save_path / 'pred_res' / f'cluster_pred_{cluster_num}_AgglomerativeClustering_{distance_strategy}{prefix}_{outlier_threshold}.npy',
        new_pred
    )


def euc_online(x, y, weight_item, center_index=None, f=None):
    """
    计算在线样本与中心样本的加权欧氏距离。
    可选写入调试信息。
    """
    euc_d_item = np.linalg.norm(x - y, axis=1) * weight_item
    if f is not None and center_index is not None:
        f.write(f"euc to {center_index}:{(euc_d_item / weight_item).tolist()}\n")
    return np.sum(euc_d_item)


class NumpyEncoder(json.JSONEncoder):
    """支持 numpy 类型序列化的 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ----------------------------
# 主流程
# ----------------------------

def main():

    # ----------------------------
    # 配置参数
    # ----------------------------
    isSmooth = True
    weight_coef = 1
    outlier_threshold = 5
    addWeight = True
    dataset_name = 'data2'
    moving_window_size = 12

    group_key = 'cluster'
    ma_size = 5
    distance_strategy = 'euc'
    cluster_dir = group_key
    prefix = "_pvt"

    # ----------------------------
    # 路径设置
    # ----------------------------
    data_root = project_path / 'code/cluster' / f'out_cluster/{dataset_name}'
    save_path = data_root / group_key
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / 'pred_res').mkdir(parents=True, exist_ok=True)
    (save_path / 'cluster_json').mkdir(parents=True, exist_ok=True)
    (save_path / f'anomaly_detection_{group_key}').mkdir(parents=True, exist_ok=True)
    (save_path / 'label_jsons').mkdir(parents=True, exist_ok=True)

    test_dataset_path = project_path / f"code/test_dataset/{dataset_name}"
    test_dataset_path.mkdir(parents=True, exist_ok=True)

    online_data_path = project_path / f"dataset/{dataset_name}/online_data.npy"
    offline_data_path = project_path / f'dataset/{dataset_name}/offline_data.npy'


    # ----------------------------
    # 加载原始数据并立即在 T 维度对齐（保留较短者，截断尾部）
    # ----------------------------
    offline_data_raw = np.load(offline_data_path)  # (N_off, T_off, F)
    online_data_raw = np.load(online_data_path)    # (N_on, T_on, F)

    assert offline_data_raw.shape[2] == online_data_raw.shape[2], \
        f"Feature dimension mismatch: offline F={offline_data_raw.shape[2]}, online F={online_data_raw.shape[2]}"

    T_off = offline_data_raw.shape[1]
    T_on = online_data_raw.shape[1]
    T_common = min(T_off, T_on)

    if T_off != T_common:
        print(f"[INFO] Truncating offline data from T={T_off} to T={T_common} (keep last {T_common} time steps)")
        offline_data_raw = offline_data_raw[:, -T_common:, :]  # keep last T_common steps

    if T_on != T_common:
        print(f"[INFO] Truncating online data from T={T_on} to T={T_common} (keep last {T_common} time steps)")
        online_data_raw = online_data_raw[:, -T_common:, :]    # keep last T_common steps

    print(f"Aligned data shapes: offline={offline_data_raw.shape}, online={online_data_raw.shape}")

    # 转置为 (N, F, T) —— 适配后续函数
    offline_data = np.transpose(offline_data_raw, (0, 2, 1))  # (N, F, T)
    online_data = np.transpose(online_data_raw, (0, 2, 1))    # (N_online, F, T)

    num_entity = offline_data.shape[0]
    time_num = offline_data.shape[2]
    feature_dim = offline_data.shape[1]

    # 保存原始在线数据（转回 T,F 供测试用）
    np.save(test_dataset_path / 'online_data.npy', online_data_raw)

    # ----------------------------
    # 平滑处理（在 F,T 上操作）
    # ----------------------------
    if isSmooth:
        moving_average(offline_data, ma_size)
        moving_average(online_data, ma_size)

    np.save(save_path / "online_data.npy", np.transpose(online_data, (0, 2, 1)))  # 保存为 (N, T, F)
    np.save(save_path / "offline_data.npy", np.transpose(offline_data, (0, 2, 1)))

    # ----------------------------
    # 离线数据预处理（需转为 T,F 格式给 preprocess 模块）
    # ----------------------------
    offline_for_preprocess = np.transpose(offline_data, (0, 2, 1))  # (N, T, F)
    np.save(save_path / "offline_data_for_preprocess.npy", offline_for_preprocess)

    param = {
        "dataset_path": save_path / "offline_data_for_preprocess.npy",
        "if_remove_extreme": True,
        "extreme_per": 0.05,
        "remove_extreme_mode": "deviation-mean",
        "if_moving_average": True,
        "moving_window_size": moving_window_size,
        "min_periods": 1,
        "if_feature_scaling": True,
        "mode": 'stand',
        "save_data_path": save_path / "preprocess_data_TF.npy",
    }
    pre.main(param)

    # 转回 (N, F, T)
    preprocess_data_TF = np.load(save_path / "preprocess_data_TF.npy")  # (N, T, F)
    preprocess_data = np.transpose(preprocess_data_TF, (0, 2, 1))       # (N, F, T)
    np.save(save_path / "preprocess_data.npy", preprocess_data)

    # ----------------------------
    # 计算周期性权重（cmndf）
    # ----------------------------
    if addWeight:
        yin_paras = {
            "data_path": save_path / 'preprocess_data.npy',
            "cycle_path": save_path / 'cycle_all.npy',
            "cmndf_path": save_path / 'cmndf.npy',
            "win_min": 96,
            "win_max": 97,
            "th": 0.3
        }
        yin.main(yin_paras)
    else:
        np.save(save_path / "cmndf.npy", np.ones((num_entity, time_num)))

    cmndf = np.load(save_path / "cmndf.npy")
    np.save(save_path / f"anomaly_detection_{group_key}/cmndf.npy", cmndf)

    # bug
    # index_to_weight_euc = 1 / (cmndf ** weight_coef)
    
    # bug fixed
    epsilon = 1e-12
    cmndf_safe = np.clip(cmndf, epsilon, None)  # 或 np.maximum(cmndf, epsilon)
    index_to_weight_euc = 1.0 / (cmndf_safe ** weight_coef)
    index_to_weight_euc = index_to_weight_euc / np.sum(index_to_weight_euc)
    
    index_to_weight_euc = np.mean(index_to_weight_euc, axis=0)
    index_to_weight_euc = index_to_weight_euc / np.sum(index_to_weight_euc)
    np.save(save_path / "index_to_weight_euc.npy", index_to_weight_euc)

    # ----------------------------
    # 构建聚类输入数据 z_to_cluster_all（直接使用整条序列）
    # ----------------------------
    data = np.load(save_path / "preprocess_data.npy")  # (N, F, T)
    np.save(save_path / 'z_to_cluster_all.npy', data)  # 不再切分！

    # ----------------------------
    # 初次计算距离矩阵
    # ----------------------------
    weight_item = np.load(save_path / "index_to_weight_euc.npy")
    data = np.load(save_path / 'z_to_cluster_all.npy')
    cal_distance(data, weight_item, distance_strategy, save_path)

    # ----------------------------
    # 相位对齐（离线）
    # ----------------------------
    euc_mat = np.load(save_path / 'euc.npy')
    s_list, aligned_data = align_phase(euc_mat, data, weight_item, save_path, prefix)

    # ----------------------------
    # 对齐后重新计算距离矩阵
    # ----------------------------
    data = np.load(save_path / f'z_to_cluster_all{prefix}.npy')
    cal_distance(data, weight_item, distance_strategy, save_path, prefix)

    # ----------------------------
    # 聚类（尝试多个簇数）  此处在entity数量（数据第0维）小于12时可能有bug
    # ----------------------------
    distance = np.load(save_path / f'{distance_strategy}{prefix}.npy')
    for cluster_num in [5, 6, 7, 8, 9]:
        print(f"Clustering with {cluster_num} clusters...")
        cluster_main(distance, cluster_num, save_path, distance_strategy, prefix, outlier_threshold)

    # ----------------------------
    # 生成聚类结构 JSON（以 cluster_num=6 为例）
    # ----------------------------
    cluster_num = 6
    cluster_pred = np.load(
        save_path / f'pred_res/cluster_pred_{cluster_num}_AgglomerativeClustering_{distance_strategy}{prefix}_{outlier_threshold}.npy'
    )
    pred_list = list(Counter(cluster_pred).items())
    print("Cluster distribution:", pred_list)

    subset_distance_matrix = np.load(save_path / f'{distance_strategy}{prefix}.npy')
    cluster_json_path = save_path / f'cluster_json/{cluster_dir}_{cluster_num}_{distance_strategy}_{outlier_threshold}.json'

    res = []
    pred_label_list = [item[0] for item in pred_list]
    for pred_label in pred_label_list:
        # 不再跳过任何标签！
        item_dict = {'label': int(pred_label), 'center': None, 'train': [], 'test': [], 'distance': []}
        item_index_list = np.where(cluster_pred == pred_label)[0]

        sub_dist_mat = subset_distance_matrix[np.ix_(item_index_list, item_index_list)]
        center_idx_local = np.argmin(np.sum(sub_dist_mat, axis=1))
        center_global = item_index_list[center_idx_local]

        distances_to_center = sub_dist_mat[center_idx_local]
        sorted_indices = np.argsort(distances_to_center)
        sorted_test = item_index_list[sorted_indices].tolist()
        sorted_distances = distances_to_center[sorted_indices].tolist()

        item_dict['center'] = int(center_global)
        item_dict['train'] = [int(center_global)]
        item_dict['test'] = sorted_test
        item_dict['distance'] = sorted_distances
        res.append(item_dict)

    with open(cluster_json_path, 'w') as f:
        json.dump(res, f)

    # ----------------------------
    # 在线数据预处理（一天数据，无需切片）
    # ----------------------------
    online_data_raw = np.load(save_path / "online_data.npy")  # (N_online, T, F)
    # 如果你确定只用一天，且 online_data 已是一天，则直接使用
    # 若原始 online_data 是多天，才需要切片，例如：
    # online_data_1day_raw = online_data_raw[:, 36*2:36*3, :]  # 取第3天
    # 但根据你的说明，已是一天，所以直接用
    online_data_1day_raw = online_data_raw

    np.save(save_path / "online_data_1day.npy", online_data_1day_raw)

    param = {
        "dataset_path": save_path / "online_data_1day.npy",
        "if_remove_extreme": True,
        "extreme_per": 0.05,
        "remove_extreme_mode": "deviation-mean",
        "if_moving_average": True,
        "moving_window_size": moving_window_size,
        "min_periods": 1,
        "if_feature_scaling": True,
        "mode": 'stand',
        "save_data_path": save_path / "preprocess_onlinedata_TF.npy",
    }
    pre.main(param)

    # 转为 (N, F, T)
    preprocess_online_TF = np.load(save_path / "preprocess_onlinedata_TF.npy")  # (N_online, T, F)
    preprocess_online = np.transpose(preprocess_online_TF, (0, 2, 1))           # (N_online, F, T)
    np.save(save_path / 'z_to_cluster_all_online.npy', preprocess_online)

    # ----------------------------
    # 在线数据相位对齐
    # ----------------------------
    weight_item = np.load(save_path / "index_to_weight_euc.npy")
    data_online = np.load(save_path / 'z_to_cluster_all_online.npy')
    data_offline = np.load(save_path / 'z_to_cluster_all_pvt.npy')
    pvt = np.argmin(np.sum(np.load(save_path / 'euc.npy'), axis=0))  # 重新获取 pvt

    s_list, aligned_online = align_phase_online(data_online, data_offline, pvt, weight_item, save_path)

    # ----------------------------
    # 在线样本分配到离线簇
    # ----------------------------
    cluster_json = json.load(open(cluster_json_path))
    z_to_cluster_all_online = np.load(save_path / "z_to_cluster_all_pvt_online.npy")
    z_to_cluster_all_offline = np.load(save_path / "z_to_cluster_all_pvt.npy")

    res = {}
    for cluster in cluster_json:
        label = cluster['label']
        center = cluster['center']
        res[label] = {"label": label, "center": center, "train": cluster['test'][:100], "test": [], "distance": []}

    for online_index in range(z_to_cluster_all_online.shape[0]):
        dis_list = [
            euc_online(z_to_cluster_all_online[online_index], z_to_cluster_all_offline[cluster['center']], weight_item)
            for cluster in cluster_json
        ]
        # print("cluster_json:", cluster_json)
        # print("len(cluster_json):", len(cluster_json))
        min_index = np.argmin(dis_list)
        cluster_label = cluster_json[min_index]['label']
        res[cluster_label]['test'].append(online_index)
        res[cluster_label]['distance'].append(float(dis_list[min_index]))

    # 排序并过滤空簇
    res_json = []
    for label, info in res.items():
        if not info['test']:
            continue
        sorted_indices = np.argsort(info['distance'])
        info['test'] = [info['test'][i] for i in sorted_indices]
        info['distance'] = [info['distance'][i] for i in sorted_indices]
        res_json.append(info)

    # 保存结果
    online_json_path = save_path / f'cluster_json/{cluster_dir}_{cluster_num}_{distance_strategy}_online_{outlier_threshold}.json'
    with open(online_json_path, 'w') as f:
        json.dump(res_json, f, cls=NumpyEncoder)

    with open(test_dataset_path / 'cluster.json', 'w') as f:
        json.dump(res_json, f, cls=NumpyEncoder)

    # ----------------------------
    # 保存测试用 offline_data（转置为 (N, T, F)）
    # ----------------------------
    data = np.load(save_path / "offline_data.npy")  # (N, T, F) —— 已保存为原始格式
    # 无需再切分，直接保存
    np.save(test_dataset_path / "offline_data.npy", data)


if __name__ == "__main__":
    main()