# OmniTransfer

## Note: Before you started, do not forget change your virtual environment.

## 0.Installation

* Python == 3.7.13
* pip install -r requirements.txt
* Notices: when testing GDN, the environment configuration is as flowing
  * Python ==3.6.13
  * conda create -n OmniTransfer_gdn_py3.6 python=3.6.13
  * pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
  * pip install torch_cluster-1.5.5-cp36-cp36m-linux_x86_64.whl
  * pip install torch_scatter-2.0.4-cp36-cp36m-linux_x86_64.whl
  * pip install torch_sparse-0.6.2-cp36-cp36m-linux_x86_64.whl
  * pip install torch_spline_conv-1.2.0-cp36-cp36m-linux_x86_64.whl
  * pip install pytorch_geometric-1.5.0.tar.gz
  * pip install tensorboard
  * pip install more_itertools

## 1.data preprocessing
Note: the NO. of SMD's entities is relatively small compared to data1 in the paper.
The shapes of train and test data are (28, T, 38), and that of label is (28, T).
28: the NO. of SMD's entities
T : time
38: features of the entities
```
python SMD/data_preprocessing.py
```

## 2.MTS clustering
Note: there is a bug in cluster.py, which is line 68.

```
python code/cluster/cluster.py --data_type=[model_name(data1 or data2)]

eg:python code/cluster/cluster_SMD.py --data_type=data1

python code/cluster/cluster_Bank_v3.py

```

## 3.Anomaly Detection
Note: there are 4 bugs in every method folder.

```
./run.sh [model_name] [out_dir]
bash ./run.sh TranAD  1029
```

## 4.adaptive transfer

Modify the parameters passed in by the adaptive transfer script **(data_type, use_center_dir_path, finetunue_all_path, freeze_rnn_path)**  to get the final results.

```
python code/transfer_eval.py --data_type=[data_type] --model_name=[model_name] --use_center_dir_path=[use_center_file_dir_path] --finetunue_all_path=[finetune_all_csv_path] --freeze_init_path=[freeze_init_csv_path]
```


```
python code/transfer_eval.py --data_type=data1 --use_center_dir_path=1028/TranAD/data1/use_center_5nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0005lr_100epoch_256bs_60ws_0.95eps --finetunue_all_path=1028/TranAD/data1/finetune_all_5nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0001lr_10epoch_256bs_60ws_0.95eps/evaluation_result/bf_machine_best_f1_g.csv --freeze_init_path=1028/TranAD/data1/freeze_att_init_last_2step_5nodes_1iwi_0.01clip_1l_8dim_1daytrain_0.0001lr_20epoch_256bs_60ws_0.95eps/evaluation_result/bf_machine_best_f1_g.csv --model_name=TranAD
```


### new pipe line
```
python run_pipeline_param.py \
  --date_offline 2021_03_05 \
  --date_online 2021_03_06 \
  --start_ts 1614972600 \
  --end_ts 1614974400 \
  --method TranAD \
  --output_folder_name 1116 \
  --output_suffix 14_to_15
```


