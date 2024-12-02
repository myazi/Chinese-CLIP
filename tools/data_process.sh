#########################################################################
# File Name: data_process.sh
# Author: yingwenjie
# mail: yingwenjie@tencent.com
# Created Time: Fri 12 Apr 2024 12:25:08 PM CST
#########################################################################
#!/bin/bash
task_name=$1
input_file=$2
model=$3

#1、将数据处理成{"text_id": 8428, "text": "高级感托特包斜挎", "image_ids": [1076345, 517602]}，这里拉图片
/apdcephfs_cq11/share_2973545/wenjieying/anaconda3/envs/python_clip/bin/python data_process.py $input_file $task_name
#2、将csv数据处理成lmbd格式，供模型读取
/apdcephfs_cq11/share_2973545/wenjieying/anaconda3/envs/python_clip/bin/python cn_clip/preprocess/build_lmdb_dataset.py \
    --data_dir data/datasets/$task_name \
    --splits train,valid,test
#3、训练模型
sh run_scripts/flickr30k_finetune_vit-b-16_rbt-base.sh data $task_name

##训练产出多个checkpoint，根据训练loss选合适的checkpoint

#4、计算测试集中keyword ann top1000数据
/apdcephfs_cq11/share_2973545/wenjieying/anaconda3/envs/python_clip/bin/python ann_res.py $input_file $task_name
#5、推理测试集中keyword-image打分，包括测试数据中高ctr、低ctr、一般数据、以及ann top1000，这里拉图片
/apdcephfs_cq11/share_2973545/wenjieying/anaconda3/envs/python_clip/bin/python clip_score.py $input_file $task_name $model
#6、根据打分取top100结果，进行可视化展示
/apdcephfs_cq11/share_2973545/wenjieying/anaconda3/envs/python_clip/bin/python clip_eval.py $task_name $model
