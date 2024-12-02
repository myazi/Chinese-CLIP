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
import hnswlib
import h5py
sys.path.append("/apdcephfs_cq11/share_2973545/wenjieying/component/Chinese-CLIP")
from ClipEmb import ClipEmb 
import os
import requests
from flask import Flask, render_template
from flask import request
import hnswlib

app = Flask(__name__)

def rerank(image_urls, index):
    image_urls_rerank = {}
    for key in image_urls:
        image_urls_rerank[key] = sorted(image_urls[key], key = lambda x:(x[index]), reverse = True)
    return image_urls_rerank 

@app.route('/')
def show_index():
    keys = []
    radio_group = request.args.get('radio_group', '2')
    radio_group = int(radio_group)
    with open("./data/0418_test.txt") as f:
        for line in f:
            key = line.strip('\n').split('\t')[3]
            if key in keys: continue
            keys.append(key)
    k = 100
    image_urls = {}
    for key in keys:
        image_urls[key] = []
    time_all = 0
    time0 = time.time()
    texts_embs = clipemb.get_text_emb(keys)
    time1 = time.time()
    faiss.ParameterSpace().set_index_parameter(faiss_index, "efSearch", int(1000))
    D, I = faiss_index.search(numpy.array(texts_embs), k)
    #I, D = faiss_index.knn_query(numpy.array(texts_embs), k)
    time2 = time.time()
    time_all += time2 - time1
    m, n = D.shape
    for i in range(m):
        for j in range(n):
            cid = I[i][j]
            url = urls[cid2index[cid]].decode('utf-8')
            print(keys[i] + "\t" + str(cid) + "\t" + url)
            #if j % 10 != 0: continue
            #cid_sta_info = cid_sta.get(cid, "-1\001-1\001-1\001-1\001-1").split("\001")
            #show = int(cid_sta_info[0])
            #click = int(cid_sta_info[1])
            #cost = int(cid_sta_info[2])
            #ctr = float(cid_sta_info[3])
            #ecpm = float(cid_sta_info[4])
            image_urls[keys[i]].append([cid, url, D[i][j]])#, show, click, cost, ctr, ecpm])
            #print(keys[i] + "\t" + str(D[0][j]) + "\t" + str(I[0][j]) + "\t" + url)
            #print(keys[i] + "\t" + str(D[0][j]) + "\t" + str(I[0][j]))
    time3 = time.time()

    print(time_all)
    print(time3 - time0)
    image_urls_rerank = rerank(image_urls, radio_group) 
    return render_template('static.html', keyword=key, image_urls=image_urls_rerank)

@app.route('/ann', methods=['GET'])
def show_ann():
    key = request.args.get('keyword',"")
    search_k = request.args.get('search_k',"")
    radio_group = request.args.get('radio_group', '2')
    radio_group = int(radio_group)
    try:
        search_k = int(search_k)
        k = search_k if search_k > 0 else 100
        print(search_k)
    except:
        k = 20 
    image_urls = {}
    image_urls[key] = []
    time_all = 0
    time0 = time.time()
    texts_embs = clipemb.get_text_emb(key)
    print(texts_embs)
    time1 = time.time()
    #faiss.ParameterSpace().set_index_parameter(faiss_index, "efSearch", int(1000))
    #D, I = faiss_index.search(numpy.array(texts_embs), k)
    I, D = faiss_index.knn_query(numpy.array(texts_embs), k)
    time2 = time.time()
    time_all += time2 - time1
    m, n = D.shape
    for i in range(m):
        for j in range(n):
            cid = I[i][j]
            url = urls[cid2index[cid]].decode('utf-8')
            #cid_sta_info = cid_sta.get(cid, "-1\001-1\001-1\001-1\001-1").split("\001")
            #show = int(cid_sta_info[0])
            #click = int(cid_sta_info[1])
            #cost = int(cid_sta_info[2])
            #ctr = float(cid_sta_info[3])
            #ecpm = float(cid_sta_info[4])
            click = key_cid.get(key, {}).get(str(cid), 0)
            image_urls[key].append([cid, url, 1 - D[i][j]]) #1 - D[i][j]])#, show, click, cost, ctr, ecpm])
            print(key + "\t" + str(cid) + "\t" + str(1 - D[i][j]))
            #print(keys[i] + "\t" + str(D[0][j]) + "\t" + str(I[0][j]) + "\t" + url)
            #print(keys[i] + "\t" + str(D[0][j]) + "\t" + str(I[0][j]))
    time3 = time.time()

    print(time_all)
    print(time3 - time0)
    image_urls_rerank = rerank(image_urls, radio_group) 
    return render_template('page.html', keyword=key, image_urls=image_urls_rerank)

@app.route('/clip', methods=['GET'])
def show_clip():
    key = request.args.get('keyword',"")
    cid = request.args.get('component_id',"")
    cid = int(cid)
    if key == "" or cid == "":
        print("key is null or cid is null")
        return
    if cid not in cid2index:
        print("cid is not in ann")
        return
    time0 = time.time()
    text_embs = clipemb.get_text_emb(key)[0]
    cid_embs = embs[cid2index[cid]]
    url = urls[cid2index[cid]].decode('utf-8')
    print(text_embs)
    print(cid_embs)
    time1 = time.time()
    clip = numpy.dot(text_embs.T, cid_embs) 
    print(clip)
    res = {}
    res[key] = [[cid, url, float(clip)]]
    return render_template('static.html', keyword=key, image_urls=res)

if __name__ == '__main__':
    global clipemb
    global faiss_index
    global cid2index
    global urls
    global cid_sta
    global key_cid

    key_cid = {}
    with open("/apdcephfs_cq11/share_2973545/wenjieying/caption/rand_key_data_sort") as f:
        for line in f:
            imp, key, cid, url, show, click, cost, all_click = line.strip('\n').split('\t')
            key_cid.setdefault(key, {})
            key_cid[key][cid] = click
    clipemb = ClipEmb()
    #faiss_index = faiss.read_index("/apdcephfs/private_wenjieying/component/ann/images_embs.index")
    #hdf5_f = h5py.File("/apdcephfs/private_wenjieying/component/ann/image_cid_url.hdf5", 'r')

    #faiss_index = faiss.read_index("/apdcephfs/private_wenjieying/component/ann/images_show_embs.index")
    #hdf5_f = h5py.File("/apdcephfs/private_wenjieying/component/ann/image_cid_url_show.hdf5", 'r')

    #faiss_index = hnswlib.Index(space = 'ip', dim = 1024)
    #faiss_index.load_index("/apdcephfs_cq11/share_2973545/wenjieying/component/data/ann/images_0408_show_embs_SQ8.index")
    #faiss_index.set_ef(1000)

    #faiss_index = faiss.read_index("/apdcephfs_cq11/share_2973545/wenjieying/component/data/ann/images_0408_old_show_embs_SQ8.index")
    #hdf5_f = h5py.File("/apdcephfs_cq11/share_2973545/wenjieying/component/data/ann/image_cid_url_0408_old_show.hdf5", 'r')

    faiss_index = hnswlib.Index(space = 'ip', dim = 1024)
    faiss_index.load_index("/apdcephfs_cq11/share_2973545/wenjieying/component/Chinese-CLIP/0128_0628_show.index")
    faiss_index.set_ef(1000)
    hdf5_f = h5py.File("/apdcephfs_cq11/share_2973545/wenjieying/component/Chinese-CLIP/0128_0628_show.hdf5", 'r')

    embs = hdf5_f["vector"][:]
    cids = hdf5_f["cid"][:]
    cids = cids.astype(int)
    urls = hdf5_f["url"][:]
    cid2index = {}
    #cid_sta = {}
    for i, cid in enumerate(cids):
        cid2index[cid] = i
    #with open("component_stat_data_month_1107") as f:
    #    for line in f:
    #        line_list = line.strip('\n').split('\t')
    #        if len(line_list) < 6: continue
    #        cid, show, click, cost, ctr, ecpm = line_list[0:7] 
    #        cid_sta[int(cid)] = "\001".join([show, click, cost, ctr, ecpm])
    app.run(host='0.0.0.0', port=4000)
