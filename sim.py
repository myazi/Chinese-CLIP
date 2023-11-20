#########################################################################
# File Name: demo.py
# Author: yingwenjie
# mail: yingwenjie@baidu.com
# Created Time: Tue 31 Oct 2023 11:24:44 AM CST
#########################################################################
import sys
import torch
import numpy
import pandas as pd
import time
import h5py

def get_sim_df(text_df, image_df, key_dict, offset):
    keys = text_df["key"]
    texts_features = numpy.vstack(text_df["emb"].to_numpy())
    print(texts_features.shape)
    cids = image_df["component_id"]
    images_features = numpy.vstack(image_df["emb"].to_numpy())
    print(images_features.shape)

    index2cid = []
    cid2index = {}
    for i in range(len(cids)):
        index2cid.append(str(cids[i]))
        cid2index[str(cids[i])] = i

    line = 0
    for key in keys:
        print(key + "@" + str(line) + "\t" + str(key_dict[key]))
        cids = []
        cid_indexs = [] 
        for cid in key_dict[key]:
            index = cid2index.get(cid, -1)
            if index == -1:
                print("miss_cid" + "\t" + cid)
                continue
            cids.append(cid)
            cid_indexs.append(index)
        cid_embs = images_features[cid_indexs]
        cosine_per_image = numpy.dot(texts_features[[line]], numpy.transpose(cid_embs))
        
        print(cosine_per_image.shape)
        n, m = cosine_per_image.shape
        for i in range(n):
            for j in range(m):
                print(key + "\t" + cids[j] + "\t" + str(cosine_per_image[i,j]))
        line += 1

def get_sim_vec(keys, texts_features, cids, images_features, key_dict):

    index2cid = []
    cid2index = {}
    for i in range(len(cids)):
        index2cid.append(str(cids[i].decode('utf-8')))
        cid2index[str(cids[i].decode('utf-8'))] = i

    line = 0
    for key in keys:
        print(key + "@" + str(line) + "\t" + str(key_dict[key]))
        cids = []
        cid_indexs = [] 
        for cid in key_dict[key]:
            index = cid2index.get(cid, -1)
            if index == -1:
                print("miss_cid" + "\t" + cid)
                continue
            cids.append(cid)
            cid_indexs.append(index)
        cid_indexs = sorted(cid_indexs)
        cid_embs = images_features[cid_indexs]
        cosine_per_image = numpy.dot(texts_features[[line]], numpy.transpose(cid_embs))
        
        print(cosine_per_image.shape)
        n, m = cosine_per_image.shape
        for i in range(n):
            for j in range(m):
                print(key + "\t" + cids[j] + "\t" + str(cosine_per_image[i,j]))
        line += 1
