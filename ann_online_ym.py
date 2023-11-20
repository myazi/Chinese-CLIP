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

app = Flask(__name__)

@app.route('/')
def show_index():
    keys = []
    with open("random_key50") as f:
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
    D, I = faiss_index.search(numpy.array(texts_embs), k)
    time2 = time.time()
    time_all += time2 - time1
    m, n = D.shape
    for i in range(m):
        for j in range(n):
            cid = I[i][j]
            url = urls[cid2index[cid]].decode('utf-8')
            cid_sta_info = cid_sta.get(cid, [-1, -1, -1, -1, -1])
            show = cid_sta_info[0]
            click = cid_sta_info[1]
            cost = cid_sta_info[2]
            ctr = cid_sta_info[3]
            ecpm = cid_sta_info[4]
            image_urls[keys[i]].append([url, D[i][j], show, click, cost, ctr, ecpm])
            #print(keys[i] + "\t" + str(D[0][j]) + "\t" + str(I[0][j]) + "\t" + url)
            #print(keys[i] + "\t" + str(D[0][j]) + "\t" + str(I[0][j]))
    time3 = time.time()

    print(time_all)
    print(time3 - time0)
    return render_template('static.html', keyword=keys, image_urls=image_urls)

@app.route('/images', methods=['GET'])
def show_images():
    key = request.args.get('keyword',"")
    k = 1000
    image_urls = {}
    image_urls[key] = []
    time_all = 0
    time0 = time.time()
    texts_embs = clipemb.get_text_emb(key)
    time1 = time.time()
    D, I = faiss_index.search(numpy.array(texts_embs), k)
    time2 = time.time()
    time_all += time2 - time1
    m, n = D.shape
    for i in range(m):
        for j in range(n):
            cid = I[i][j]
            url = urls[cid2index[cid]].decode('utf-8')
            cid_sta_info = cid_sta.get(cid, [-1, -1, -1, -1, -1])
            show = cid_sta_info[0]
            click = cid_sta_info[1]
            cost = cid_sta_info[2]
            ctr = cid_sta_info[3]
            ecpm = cid_sta_info[4]
            image_urls[key].append([url, D[i][j], show, click, cost, ctr, ecpm])
            #print(keys[i] + "\t" + str(D[0][j]) + "\t" + str(I[0][j]) + "\t" + url)
            #print(keys[i] + "\t" + str(D[0][j]) + "\t" + str(I[0][j]))
    time3 = time.time()

    print(time_all)
    print(time3 - time0)
    return render_template('static.html', keyword=key, image_urls=image_urls)

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
    faiss_index = faiss.read_index("/apdcephfs_cq3/share_2973545/wenjieying/component/data/ann/images_show_embs.index")
    hdf5_f = h5py.File("/apdcephfs_cq3/share_2973545/wenjieying/component/data/ann/image_cid_url_show.hdf5", 'r')
    cids = hdf5_f["cid"][:]
    cids = cids.astype(int)
    urls = hdf5_f["url"][:]
    cid2index = {}
    cid_sta = {}
    i = 0 
    for cid in cids:
        cid2index[cid] = i
        i += 1
    with open("component_stat_data_month_1107") as f:
        for line in f:
            line_list = line.strip('\n').split('\t')
            if len(line_list) < 6: continue
            cid, show, click, cost, ctr, ecpm = line_list[0:7] 
            cid_sta[int(cid)] = [int(show), int(click), int(cost), round(float(ctr), 4), round(float(ecpm), 4)]
    app.run(host='0.0.0.0', port=80)
