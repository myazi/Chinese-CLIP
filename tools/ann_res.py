#########################################################################
# File Name: clip_eval.py
# Author: yingwenjie
# mail: yingwenjie@tencent.com
# Created Time: Wed 03 Apr 2024 05:19:51 PM CST
#########################################################################
import sys
import os
import hnswlib
import h5py
import numpy
import pandas as pd
from ClipEmb import ClipEmb 

def ann_index():
    faiss_index = hnswlib.Index(space = 'ip', dim = 1024)
    faiss_index.load_index("cid_0230_0617.index")
    faiss_index.set_ef(1000)
    hdf5_f = h5py.File("cid_0230_0617.hdf5", 'r')
    return faiss_index, hdf5_f

def get_data(file_name):
    key_cids = {}
    df = pd.read_csv(file_name, sep='\t')
    keys = df['keyword']
    #cids = df['component_id']
    cids = df['keyword']
    for key, cid in zip(keys, cids):
        key_cids.setdefault(key, [])
        key_cids[key].append(cid)
    return key_cids

def ann_res(faiss_index, hdf5_f, clipemb, key_cids, ann_file):
    k = 1000
    cids = hdf5_f["cid"]
    urls = hdf5_f["url"]
    cid_url = {}
    for cid, url in zip(cids, urls):
        cid_url[cid] = url.decode('utf-8')
    out = open(ann_file, 'w')
    out.write("keyword\tcomponent_id\timage_url\tsim\tflag\n")
    for keyword in key_cids:
        cids = key_cids[keyword]
        texts_embs = clipemb.get_text_emb(keyword)
        I, D = faiss_index.knn_query(numpy.array(texts_embs), k)
        m, n = D.shape
        for i in range(m):
            for j in range(n):
                cid = I[i][j]
                url = cid_url.get(cid)
                flag = 0
                if str(cid) in cids:
                    flag = 1
                out.write(keyword + "\t" + str(I[i][j]) + "\t" + str(url) + "\t" + str(1 - D[i][j]) + "\t"  + str(flag) + "\n")

if __name__ == '__main__':
    data_file = sys.argv[1]
    dataset_name = sys.argv[2]
    dataset_dir = "/apdcephfs_cq11/share_2973545/wenjieying/component/Chinese-CLIP/data/test"
    dataset_dir = os.path.join(dataset_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    ann_file = os.path.join(dataset_dir, dataset_name + "_ann_res")
    key_cids = get_data(data_file)
    print(len(key_cids))
    model_path = "/apdcephfs_cq11/share_2973545/wenjieying/component/Chinese-CLIP/data/experiments/0125_0611_sort_clip_cn_vit-h-14/checkpoints/epoch1.pt"
    clipemb = ClipEmb(model_path=model_path)
    faiss_index, hdf5_f = ann_index()
    ann_res(faiss_index, hdf5_f, clipemb, key_cids, ann_file)
