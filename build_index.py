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

def get_df(df_files, tp):
    out_file = "image_cid_url.hdf5"
    time1 = time.time()
    if tp == "text":
        pds = [pd.read_parquet(df_file) for df_file in df_files]
        df = pd.concat(pds, ignore_index=True)
        time3 = time.time()
        print("read df" + str(time3 - time2))
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

def get_h5_sub(h5_file, out_file, cid_file):
    cids = set()
    with open(cid_file) as f:
        for line in f:
            cid = line.strip('\n')
            cids.add(int(cid))
    print(len(cids))
    hdf5_f = h5py.File(h5_file, 'r')
    h5_vector = hdf5_f["vector"][:]
    h5_cids = hdf5_f["cid"][:]
    h5_cids = h5_cids.astype(int)
    h5_urls = hdf5_f["url"][:]
    subs = [cid for cid in h5_cids if cid in cids]
    print(len(subs))
    embs = h5_vector[subs]
    print(embs.shape)
    print(embs[0])
    cids = h5_cids[subs]
    urls = h5_urls[subs]
    f = h5py.File(out_file, 'w')
    f.create_dataset("vector", data=embs)
    f.create_dataset("cid", data=cids)
    f.create_dataset("url", data=urls)
    f.close()

def build_index(images_embs, cids, urls, texts_embs)
    dim = 1024
    m = 32
    efc = 500
    efs = 800
    k = 10
    qtype = getattr(faiss.ScalarQuantizer, "QT_8bit")
    faiss_index = faiss.IndexHNSWSQ(dim, qtype, m, faiss.METRIC_INNER_PRODUCT)
    faiss_index.verbose = True
    faiss_index.hnsw.efConstruction = efc
    faiss_index.hnsw.efSearch = efs
    faiss_index = faiss.IndexIDMap(faiss_index)
    faiss_index.train(images_embs)
    faiss_index.add_with_ids(images_embs, cids)
    faiss.write_index(faiss_index, "images_show_embs.index")
    for i in range(10):
        D, I = faiss_index.search(numpy.array([texts_embs[i]]), k)
        print(keys[i] + "\t" + str(D) + "\t" + str(I))

def build_index_hnswlib(images_embs, cids, urls, texts_embs)
    dim = 1024
    m = 32
    efc = 800
    efs = 1000
    k = 10
    hnsw_index = hnswlib.Index(space='ip', dim=len(images_embs[0]))
    hnsw_index.init_index(max_elements=len(images_embs),
            ef_construction=efc,
            M=m)
    hnsw_index.add_items(images_embs, cids)
    hnsw_index.save_index("images_show_embs_hnswlib.index")
    print("index train done")
    hnsw_index.set_ef(efs)
    for i in range(10):
        D, I = hnsw_index.knn_query(numpy.array([texts_embs[i]]), k)
        print(keys[i] + "\t" + str(D) + "\t" + str(I))


if __name__ == '__main__':

    #1 get texts and images df, get key_cids images.hdf5
    texts = ["/apdcephfs_cq3/share_2973545/wenjieying/component/data/texts_vec_1107/text_" + str(i).zfill(5) + ".parquet" for i in range(1,2)]
    df_text = get_df(texts, "text") 
    #images = ["/apdcephfs_cq3/share_2973545/wenjieying/component/data/images_vec/image_" + str(i).zfill(5) + ".parquet" for i in range(1,1161)]
    #df = get_df(images, "image")

    #2 from images.hdf5 select images_show to image_show.hdf5
    #get_h5_sub("../ann/image_cid_url.hdf5", "image_cid_url_show.hdf5", "../show_month_1107_all_cuid")
    #exit()

    #3 from hdf5 get embedding build index
    texts_embs = numpy.vstack(df_text["emb"].to_numpy())
    keys = df_text["key"]
    hdf5_f = h5py.File("image_cid_url_show.hdf5", 'r')
    images_embs = hdf5_f["vector"]
    cids = hdf5_f["cid"][:]
    cids = cids.astype(int)
    urls = hdf5_f["url"]
    print(texts_embs.shape)
    print(images_embs.shape)
    print(cids.shape)
    build_index(images_embs, cids, urls, texts_embs)

