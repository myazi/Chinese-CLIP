#########################################################################
# File Name: clip_eval.py
# Author: yingwenjie
# mail: yingwenjie@tencent.com
# Created Time: Fri 12 Apr 2024 07:34:45 PM CST
#########################################################################
import sys
import os
import pandas as pd

def cal_recall(file_name, out_file="", topk = 100):
    df = pd.read_csv(file_name, sep='\t')
    keys = df['keyword']
    cids = df['component_id']
    urls = df['image_url']
    tags = df['tag']
    scores = df['sim']
    key_res = {}
    for keyword, cid, url, tag, score in zip(keys, cids, urls, tags, scores):
        key_res.setdefault(keyword,[])
        key_res[keyword].append([cid, url, tag, score])

    recall_ann = 0
    recall_0 = 0
    pre_pos = 0
    pre_neg = 0
    recall_pos = 0
    recall_neg = 0
    all_num = len(key_res) * topk
    all_pos = 0 
    all_neg = 0
    out = open(out_file, 'w')
    out.write("keyword\tcomponent_id\timage_url\ttag\tsim\n")
    for key in key_res:
        res_list = key_res[key]
        res_list_sort = sorted(res_list, key = lambda x:x[3], reverse=True)
        k = topk
        for res in res_list_sort:
            k -= 1
            tag = res[2]
            if tag == "-1": all_neg += 1
            if tag == "1": all_pos += 1
            if k < 0: continue
            if tag == "-1": recall_neg += 1
            if tag == "0": recall_0 += 1
            if tag == "1": recall_pos += 1
            if tag == "ann": recall_ann += 1
            out.write(key + "\t" + "\t".join([str(i) for i in res]) + "\n")
    recall_ann /= all_num
    recall_0 /= all_num
    pre_pos = recall_pos / all_num
    pre_neg = recall_neg / all_num
    recall_pos /= all_pos
    recall_neg /= all_neg
    recall_ann = round(recall_ann, 4)
    recall_0 = round(recall_0, 4)
    recall_pos = round(recall_pos, 4)
    recall_neg = round(recall_neg, 4)
    pre_pos = round(pre_pos, 4)
    pre_neg = round(pre_neg, 4)
    

    print("recall_pos\trecall_neg\tpre_pos\tpre_neg\trecall_0\trecall_ann")
    print(str(recall_pos) + "\t" + str(recall_neg) + "\t" + str(pre_pos) + "\t" + str(pre_neg) + "\t" + str(recall_0) + "\t" + str(recall_ann))
    #out.write("recall_pos\trecall_neg\tpre_pos\tpre_neg\trecall_0\trecall_ann\n")
    #out.write(str(recall_pos) + "\t" + str(recall_neg) + "\t" + str(pre_pos) + "\t" + str(pre_neg) + "\t" + str(recall_0) + "\t" + str(recall_ann) + "\n")

if __name__ == '__main__':
    dataset_name = sys.argv[1]
    model_name = sys.argv[2]
    dataset_dir = "/apdcephfs_cq11/share_2973545/wenjieying/component/Chinese-CLIP/data/test"
    dataset_dir = os.path.join(dataset_dir, dataset_name)
    res_file = os.path.join(dataset_dir, dataset_name + "_score_" + model_name)
    out_file = os.path.join(dataset_dir, dataset_name + "_rank_" + model_name)
    cal_recall(res_file, out_file)
