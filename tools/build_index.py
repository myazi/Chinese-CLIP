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
import hnswlib

def dot_product(v1, v2):
    matrix = numpy.dot(v1, v2)
    return numpy.diagonal(matrix)

def vector_norm(p_rep):
    p_rep = numpy.asarray(p_rep)
    p_rep = p_rep.reshape(1, 64)
    p_vector = dot_product(p_rep ,p_rep.T)
    norm = numpy.sqrt(p_vector)
    return p_rep / norm.reshape(-1, 1)

def get_line(file_name, out_file):
    cid_limit = set()
    with open("/apdcephfs_cq11/share_2973545/wenjieying/metapath2vec/bin/key_id5_ann_index") as f:
        for line in f:
            cid = line.strip('\n')
            cid_limit.add(cid)

    cids = []
    embs = numpy.zeros((3290395, 64), dtype="float16") #19866649
    i = 0
    with open(file_name) as f:
        for line in f:
            line_list = line.strip('\n').split(' ')
            if len(line_list) != 66: continue
            key = line_list[0]
            if 'a' not in key: continue
            #if key not in cid_limit: continue
            key_a = key.replace('a','')
            vecs = [float(i) for i in line_list[1:-1]]
            embs[i:i+1, :] = vector_norm(numpy.asarray(vecs, dtype=numpy.float16))
            cids.append(key_a)
            i+= 1
            if i % 10000 == 0:
                print(key_a)
                print(i)
    print(key_a)
    print(len(cids))
    print(embs.shape)
    f = h5py.File(out_file, 'w')
    f.create_dataset("vector", data=embs)
    f.create_dataset("cid", data=cids)
    f.close()


def get_df(df_files, tp, out_file=""):
    time1 = time.time()
    if tp == "text":
        time2 = time.time()
        pds = [pd.read_parquet(df_file) for df_file in df_files]
        df = pd.concat(pds, ignore_index=True)
        time3 = time.time()
        print("read df" + str(time3 - time2))
        return df
    if tp == "image":
        cids = []
        urls = []
        embs = numpy.zeros((1270179, 1024), dtype="float16") #19866649
        #embs = numpy.zeros((100000, 1024), dtype="float16") #19866649
        start = 0
        for i, df_file in zip(range(len(df_files)), df_files):
            print(df_file)
            ppd = pd.read_parquet(df_file, columns=['component_id', 'url', 'emb'])
            ppd = ppd[ppd['emb'].notnull()]
            emb = numpy.vstack(ppd["emb"].to_numpy()) 
            length = emb.shape[0]
            num_rows, num_columns = ppd.shape
            print(num_rows)
            embs[start : start + num_rows, :] = numpy.vstack(ppd["emb"].to_numpy())
            for cid in ppd['component_id']: cids.append(cid)
            for url in ppd['url']: urls.append(url)
            start += num_rows 
            print(start)
        #cids = numpy.array(cids, dtype='S')
        urls = numpy.array(urls, dtype='S')
        f = h5py.File(out_file, 'w')
        f.create_dataset("vector", data=embs)
        f.create_dataset("cid", data=cids)
        f.create_dataset("url", data=urls)
        f.close()

def get_h5_show(cid_file, h5_file, out_file):
    cids = set()
    with open(cid_file, encoding='utf-8-sig') as f:
        for line in f:
            cid = line.strip('\n').split('\t')[0]
            if cid == "component_id": continue
            cids.add(int(cid))
    print(len(cids))
    hdf5_f = h5py.File(h5_file, 'r')
    h5_vector = hdf5_f["vector"][:]
    h5_cids = hdf5_f["cid"][:]
    h5_cids = h5_cids.astype(int)
    h5_urls = hdf5_f["url"][:]
    subs = [i for i, cid in zip(range(len(h5_cids)), h5_cids) if cid in cids]
    print(len(subs))
    embs = h5_vector[subs]
    cids = h5_cids[subs]
    urls = h5_urls[subs]
    f = h5py.File(out_file, 'w')
    f.create_dataset("vector", data=embs)
    f.create_dataset("cid", data=cids)
    f.create_dataset("url", data=urls)
    f.close()

def index_type_pca(pca, dim=1024):
    pca_str = "PCA" + str(pca) + ",Flat"
    faiss_index = faiss.index_factory(dim, pca_str, faiss.METRIC_INNER_PRODUCT)
    faiss_index = faiss.IndexIDMap(faiss_index)
    return faiss_index

def index_type_pq(m=32, nbits=8, dim=1024):                                                                                                                                              
    faiss_index = faiss.IndexPQ(dim, m, nbits)
    faiss_index = faiss.IndexIDMap(faiss_index)
    return faiss_index

def build_index_Flat(images_embs, cids, urls, texts_embs, keys):
    dim = 64
    #num = faiss.omp_get_max_threads()
    #faiss.omp_set_num_threads(num)
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index = faiss.IndexIDMap(faiss_index)
    faiss_index.add_with_ids(images_embs, cids)
    #for cid, image_embs in zip(cids, images_embs):
    #    print(str(cid) + "\t" + " ".join([str(i) for i in image_embs]))
    faiss.write_index(faiss_index, "images_0408_show_embs_Flat.index")


def build_index(images_embs, cids, urls, texts_embs, keys):
    dim = 1024
    m = 24
    efc = 500
    efs = 1000
    k = 100
    num = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(num)
    qtype = getattr(faiss.ScalarQuantizer, "QT_8bit")
    faiss_index = faiss.IndexHNSWSQ(dim, qtype, m, faiss.METRIC_INNER_PRODUCT)
    faiss_index.verbose = True
    faiss_index.hnsw.efConstruction = efc
    faiss_index.hnsw.efSearch = efs
    #faiss_index = faiss.IndexIDMap(faiss_index)
    #faiss_index = index_type_pca(32)
    #faiss_index = index_type_pq()
    faiss_index.train(images_embs)
    #faiss_index.add_with_ids(images_embs, cids)
    faiss_index.add(images_embs)
    faiss.write_index(faiss_index, "all_keys_25M.index_M24_500")
    exit()
    for l in range(100):
        D, I = faiss_index.search(numpy.array([texts_embs[l]]), k)
        m, n = D.shape
        for i in range(m):
            for j in range(n):
                cid = I[i][j]
                print(keys[l] + "\t" + str(I[i][j]) + "\t" + str(D[i][j]))

def build_index_hnswlib(images_embs, cids, urls, texts_embs, keys):
    m = 32
    efc = 800
    efs = 1000
    k = 100
    hnsw_index = hnswlib.Index(space='ip', dim=len(images_embs[0]))
    hnsw_index.init_index(max_elements=len(images_embs),
            ef_construction=efc,
            M=m)
    print("start train")
    hnsw_index.add_items(images_embs, cids)
    hnsw_index.save_index("20240829_cids_YouClip_base.index")
    print("index train done")
    exit()
    hnsw_index.set_ef(efs)
    for l in range(154):
        I, D = hnsw_index.knn_query(numpy.array([texts_embs[l]]), k)
        m, n = D.shape
        for i in range(m):
            for j in range(n):
                cid = I[i][j]
                print(keys[l] + "\t" + str(I[i][j]) + "\t" + str(1 - D[i][j]))

if __name__ == '__main__':

    #get_line("/apdcephfs_cq11/share_2973545/wenjieying/metapath2vec/bin/out_train_key_image_seq5_key.txt", "/apdcephfs_cq11/share_2973545/wenjieying/metapath2vec/bin/out_train_key_image_seq5_key.hdf5")
    #exit()
    #data_dir = sys.argv[1]
    #h5_file = sys.argv[2] #image_cid_url_0408.hdf5
    #1 get texts and images df, get key_cids images.hdf5
    #texts = [data_dir + "/text_" + str(i).zfill(5) + ".parquet" for i in range(0,1)]
    #df_text = get_df(texts, "text") 
    
    #2
    #images = [data_dir + "/m2_encoder_1B_" + str(i).zfill(5) + ".parquet" for i in range(0, 129)]
    #print(images)
    #get_df(images, "image", h5_file)
    #exit()

    #2 from images.hdf5 select images_show to image_show.hdf5
    #get_h5_show("./data/image_20240408_post_20240108_20240408_show", "image_cid_url_0408.hdf5", "image_cid_url_show.hdf5")
    #get_h5_show("./data/image_cid_url_show_cid", "image_cid_url_0408.hdf5", "image_cid_url_old_show.hdf5")
    #get_h5_show("./data/test/0418_test/0418_test_image_file", "image_cid_url_0408.hdf5", "image_cid_url_test_show.hdf5")
    #exit()

    #3 from hdf5 get embedding build index
    #texts_embs = numpy.vstack(df_text["emb"].to_numpy())
    #keys = df_text["key"]
    #texts_embs = df_text["emb"]

    h5_file = "/apdcephfs_cq11/share_2973545/wenjieying/metapath2vec/bin/out_train_key_image_seq5_key_click_10.hdf5"
    h5_file = "cid_0230_0617_epoch10.hdf5"
    h5_file = "0302_0602_click10.hdf5"
    h5_file = "cid_click_xq_20240528_20240628_click10.hdf5"
    h5_file = "20240829_cids_m2_1B.hdf5"
    #h5_file = "20240829_cids_ernie.hdf5"
    h5_file = "all_keys_25M.hdf5"
    h5_file = "20240829_cids_YouClip_base.hdf5"
    hdf5_f = h5py.File(h5_file, 'r')
    images_embs = hdf5_f["vector"]
    #images_embs = images_embs[:10000]
    cids = hdf5_f["cid"][:]
    #cids = cids.astype(int)
    #urls = hdf5_f["url"]
    urls = ""
    texts_embs = ""
    keys = ""
    #print(texts_embs.shape)
    print(images_embs.shape)
    print(cids.shape)
    build_index_hnswlib(images_embs, cids, urls, texts_embs, keys)
    #build_index(images_embs, cids, urls, texts_embs, keys)
    #build_index_Flat(images_embs, cids, urls, texts_embs, keys)
