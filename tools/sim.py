#########################################################################
# File Name: demo.py
# Author: yingwenjie
# mail: yingwenjie@baidu.com
# Created Time: Tue 31 Oct 2023 11:24:44 AM CST
#########################################################################
import sys
import os
import numpy
import pandas as pd

key_set = set()
cid_set = set()
key_embs = {}
cid_embs = {}
def get_data():
    with open("./data/user_action_url_0125_0621_sort") as f:
        i = 0
        for line in f:
            i += 1
            #if i > 1000000: break
            line_list = line.strip('\n').split('\t')
            if len(line_list) < 2: continue
            key, cid = line_list[0:2]
            key_set.add(key)
            cid_set.add(cid)
    print(len(key_set))
    print(len(cid_set))

def get_sim_text(text_file):
    text_df= pd.read_parquet(text_file, columns=['key', 'emb'])
    keys = text_df["key"]
    texts_features = numpy.vstack(text_df["emb"].to_numpy())
    print(texts_features.shape)
    for key, emb in zip(keys, texts_features):
        if key in key_set:
            key_embs[key] = numpy.array(emb)
    print(len(key_embs))

def get_sim_image(image_file):
    image_df= pd.read_parquet(image_file, columns=['component_id', 'emb'])
    cids = image_df["component_id"]
    images_features = numpy.vstack(image_df["emb"].to_numpy())
    print(images_features.shape)
    for cid, emb in zip(cids, images_features):
        cid = str(cid)
        if cid in cid_set:
            cid_embs[cid] = numpy.array(emb)
    print(len(cid_embs))

def cal_sim(key_embs, cid_embs):
    with open("./data/user_action_url_0125_0621_sort") as f:
        for line in f:
            line_list = line.strip('\n').split('\t')
            if len(line_list) < 2: continue
            key, cid = line_list[0:2]
            key_emb = key_embs.get(key, "")
            cid_emb = cid_embs.get(cid, "")
            if key_emb != "" and cid_emb != "":
                sim = numpy.dot(key_emb.T, cid_emb) 
                print(line.strip('\n') + "\t" + str(round(sim, 4)))
            else:
                print("nono")

if __name__ == '__main__':
    get_data()
    text_dir = "/apdcephfs_cq11/share_2973545/image_generation/component/keyword/keyword_history_clip"
    image_dir = "/apdcephfs_cq11/share_2973545/image_generation/component/image/image_history_clip"
    texts = os.listdir(text_dir) 
    images = os.listdir(image_dir) 
    print(texts)
    print(images)
    i = 0
    for image_df in images: 
        i += 1
        #if i > 10: break
        get_sim_image(os.path.join(image_dir, image_df))
    i = 0
    for text_df in texts: 
        i += 1
        #if i > 10: break
        get_sim_text(os.path.join(text_dir, text_df))
    
    cal_sim(key_embs, cid_embs)
