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
    radio_group = request.args.get('radio_group', '1')
    radio_group = int(radio_group)
    with open("./tmp/random_key50") as f:
        for line in f:
            keys.append(line.strip('\n'))
    k = 1000
    image_urls = {}
    for key in keys:
        image_urls[key] = []
    time_all = 0
    time0 = time.time()
    texts_embs = clipemb.get_text_emb(keys)
    time1 = time.time()
    #D, I = faiss_index.search(numpy.array(texts_embs), k)
    I, D = faiss_index.knn_query(numpy.array(texts_embs), k)
    time2 = time.time()
    time_all += time2 - time1
    m, n = D.shape
    for i in range(m):
        for j in range(n):
            if j % 10 != 0: continue
            cid = I[i][j]
            url = urls[cid2index[cid]].decode('utf-8')
            cid_sta_info = cid_sta.get(cid, "-1\001-1\001-1\001-1\001-1").split("\001")
            show = int(cid_sta_info[0])
            click = int(cid_sta_info[1])
            cost = int(cid_sta_info[2])
            ctr = float(cid_sta_info[3])
            ecpm = float(cid_sta_info[4])
            image_urls[keys[i]].append([url, 1 - D[i][j], show, click, cost, ctr, ecpm])
            #print(keys[i] + "\t" + str(D[0][j]) + "\t" + str(I[0][j]) + "\t" + url)
            #print(keys[i] + "\t" + str(D[0][j]) + "\t" + str(I[0][j]))
    time3 = time.time()

    print(time_all)
    print(time3 - time0)
    image_urls_rerank = rerank(image_urls, radio_group) 
    return render_template('static.html', keyword=key, image_urls=image_urls_rerank)

@app.route('/images', methods=['GET'])
def show_images():
    key = request.args.get('keyword',"")
    radio_group = request.args.get('radio_group', '1')
    radio_group = int(radio_group)
    k = 1000
    image_urls = {}
    image_urls[key] = []
    time_all = 0
    time0 = time.time()
    texts_embs = clipemb.get_text_emb(key)
    time1 = time.time()
    #D, I = faiss_index.search(numpy.array(texts_embs), k)
    I, D = faiss_index.knn_query(numpy.array(texts_embs), k)
    time2 = time.time()
    time_all += time2 - time1
    m, n = D.shape
    for i in range(m):
        for j in range(n):
            cid = I[i][j]
            url = urls[cid2index[cid]].decode('utf-8')
            cid_sta_info = cid_sta.get(cid, "-1\001-1\001-1\001-1\001-1").split("\001")
            show = int(cid_sta_info[0])
            click = int(cid_sta_info[1])
            cost = int(cid_sta_info[2])
            ctr = float(cid_sta_info[3])
            ecpm = float(cid_sta_info[4])
            image_urls[key].append([url, 1- D[i][j], show, click, cost, ctr, ecpm])
            #print(keys[i] + "\t" + str(D[0][j]) + "\t" + str(I[0][j]) + "\t" + url)
            #print(keys[i] + "\t" + str(D[0][j]) + "\t" + str(I[0][j]))
    time3 = time.time()

    print(time_all)
    print(time3 - time0)
    image_urls_rerank = rerank(image_urls, radio_group) 
    return render_template('static.html', keyword=key, image_urls=image_urls_rerank)

if __name__ == '__main__':
    global clipemb
    global faiss_index
    global cid2index
    global urls
    global cid_sta

    clipemb = ClipEmb()
    #faiss_index = faiss.read_index("/apdcephfs/private_wenjieying/component/ann/images_embs.index")
    #hdf5_f = h5py.File("/apdcephfs/private_wenjieying/component/ann/image_cid_url.hdf5", 'r')

    #faiss_index = faiss.read_index("/apdcephfs/private_wenjieying/component/ann/images_show_embs.index")
    #hdf5_f = h5py.File("/apdcephfs/private_wenjieying/component/ann/image_cid_url_show.hdf5", 'r')

    faiss_index = hnswlib.Index(space = 'ip', dim = 1024)
    faiss_index.load_index("/apdcephfs/private_wenjieying/component/ann/images_show_embs_hnswlib.index")
    faiss_index.set_ef(1000)
    hdf5_f = h5py.File("/apdcephfs/private_wenjieying/component/ann/image_cid_url_show.hdf5", 'r')

    cids = hdf5_f["cid"][:]
    cids = cids.astype(int)
    urls = hdf5_f["url"][:]
    cid2index = {}
    cid_sta = {}
    for i, cid in zip(range(len(cids)), cids):
        cid2index[cid] = i
    with open("component_stat_data_month_1107") as f:
        for line in f:
            line_list = line.strip('\n').split('\t')
            if len(line_list) < 6: continue
            cid, show, click, cost, ctr, ecpm = line_list[0:7] 
            cid_sta[int(cid)] = "\001".join([show, click, cost, ctr, ecpm])
    app.run(host='0.0.0.0', port=3000)
