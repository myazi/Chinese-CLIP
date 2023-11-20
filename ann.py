#########################################################################
# File Name: demo.py
# Author: yingwenjie
# mail: yingwenjie@baidu.com
# Created Time: Tue 31 Oct 2023 11:24:44 AM CST
#########################################################################
import sys
import numpy
import pandas as pd
import time
import faiss
import h5py
from ClipEmb import ClipEmb 
import hnswlib

def get_df(df_files, tp):
    out_file = "image_cid_url.hdf5"
    time1 = time.time()
    if tp == "text":
        pds = [pd.read_parquet(df_file) for df_file in df_files]
        df = pd.concat(pds, ignore_index=True)
        time2 = time.time()
        print("read df" + str(time2 - time1))
        return df
    if tp == "image":
        embs = numpy.zeros((11595676, 1024), dtype="float16")
        cids = []
        urls = []
        for i, df_file in zip(range(len(df_files)), df_files):
            ppd = pd.read_parquet(df_file, columns=['component_id', 'url', 'emb'])
            emb = numpy.vstack(ppd["emb"].to_numpy()) 
            length = emb.shape[0]
            embs[i * 10000 : i * 10000 + min(10000, length), :] = numpy.vstack(ppd["emb"].to_numpy())
            for cid in ppd['component_id']: cids.append(cid)
            for url in ppd['url']: urls.append(url)
            print(i)
        cids = numpy.array(cids, dtype='S')
        urls = numpy.array(urls, dtype='S')
        f = h5py.File(out_file, 'w')
        f.create_dataset("vector", data=embs)
        f.create_dataset("cid", data=cids)
        f.create_dataset("url", data=urls)
        f.close()
    time2 = time.time()
    print("read df" + str(time2 - time1))
    return df

def ann_search_one():
    keys = []
    with open("one_key") as f:
        for line in f:
            key = line.strip('\n').split('\t')[0]
            keys.append(key)
    hdf5_f = h5py.File("../data/ann/image_cid_url.hdf5", 'r')
    cids = hdf5_f["cid"][:]
    cids = cids.astype(int)
    urls = hdf5_f["url"]
    cid2index = {}
    for i, cid in zip(range(len(cids)) ,cids):
        cid2index[cid] = i

    clipemb = ClipEmb()
    texts_embs = clipemb.get_text_emb(keys)
    texts_embs.tolist()
    for text_embs in texts_embs:
        print(" ".join([ str(i) for i in text_embs]))

    faiss_index = faiss.read_index("../data/ann/images_embs.index")
    faiss.ParameterSpace().set_index_parameter(faiss_index, "efSearch", 1000)
    k = 1000
    D, I = faiss_index.search(numpy.array(texts_embs), k)
    m, n = D.shape
    for i in range(m):
        for j in range(n):
            print(keys[i] + "\t" + str(D[i][j]) + "\t" + str(I[i][j]))

def ann_search():

    #texts = ["/apdcephfs_cq3/share_2973545/wenjieying/component/texts_vec_1107/text_" + str(i).zfill(5) + ".parquet" for i in range(1,544)]
    #texts = ["/apdcephfs_cq3/share_2973545/wenjieying/component/text_" + str(i).zfill(5) + ".parquet" for i in range(1,1307)]
    texts = ["/apdcephfs_cq3/share_2973545/wenjieying/component/data/texts_vec_1107_show_1053_500/text_" + str(i).zfill(5) + ".parquet" for i in range(1,1055)]
    texts_out = ["/apdcephfs_cq3/share_2973545/wenjieying/component/data/texts_vec_1107_show_1053_500_ann/text_" + str(i).zfill(5) + "_ann" for i in range(1,1055)]

    faiss_index = faiss.read_index("../data/ann/images_show_embs.index")
    faiss.ParameterSpace().set_index_parameter(faiss_index, "efSearch", 1000)
    k = 1000

    for t in range(len(texts)):
        text = texts[t]
        df_text = get_df([text], "text") 
        texts_embs = numpy.vstack(df_text["emb"].to_numpy())
        keys = df_text["key"]

        D, I = faiss_index.knn_query(numpy.array(texts_embs), k)
        res = []
        m, n = D.shape
        for i in range(m):
            for j in range(n):
                res.append(keys[i] + "\t" + str(I[i][j]) + "\t" + str(D[i][j])) 
        text_out = texts_out[t]
        f = open(text_out, 'w')
        f.write("\n".join(res))
        f.close()

def ann_search_hnswlib():

    #texts = ["/apdcephfs_cq3/share_2973545/wenjieying/component/texts_vec_1107/text_" + str(i).zfill(5) + ".parquet" for i in range(1,544)]
    #texts = ["/apdcephfs_cq3/share_2973545/wenjieying/component/text_" + str(i).zfill(5) + ".parquet" for i in range(1,1307)]
    texts = ["/apdcephfs_cq3/share_2973545/wenjieying/component/data/texts_vec_1107_show_1053_500_2/text_" + str(i).zfill(5) + ".parquet" for i in range(1,501)]
    texts_out = ["/apdcephfs_cq3/share_2973545/wenjieying/component/data/texts_vec_1107_show_1053_500_ann_2/text_" + str(i).zfill(5) + "_ann" for i in range(1,501)]

    faiss_index = hnswlib.Index(space = 'ip', dim = 1024)
    faiss_index.load_index("../data/ann/images_show_embs_hnswlib.index")
    faiss_index.set_ef(1000)
    k = 1000

    for t in range(len(texts)):
        text = texts[t]
        df_text = get_df([text], "text") 
        texts_embs = numpy.vstack(df_text["emb"].to_numpy())
        keys = df_text["key"]

        I, D = faiss_index.knn_query(numpy.array(texts_embs), k)
        res = []
        m, n = D.shape
        for i in range(m):
            for j in range(n):
                res.append(keys[i] + "\t" + str(I[i][j]) + "\t" + str(1 - D[i][j])) 
        text_out = texts_out[t]
        f = open(text_out, 'w')
        f.write("\n".join(res))
        f.close()

if __name__ == '__main__':
    ann_search_hnswlib()

