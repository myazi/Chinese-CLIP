#########################################################################
# File Name: data_process.py
# Author: yingwenjie
# mail: yingwenjie@tencent.com
# Created Time: Wed 03 Apr 2024 02:20:40 PM CST
#########################################################################
import sys
import math
from PIL import Image
from io import BytesIO
import json
import pandas as pd
import base64
from img2dataset import download
import os
import random

def data_sort():
    keys = {}
    cids = {}
    i = 0
    with open ("data/user_action_url_0125_0621_sort_sim_select2") as f:
        for line in f:
            i += 1
            if i == 1: continue
            line_list = line.strip('\n').split('\t')
            if len(line_list) < 7: continue
            keyword, cid, url, show, click, cost, gmv = line_list[0:7]
            value = "\t".join(line_list[1:7])
            show = int(show)
            click = int(click)
            cost = float(cost)
            gmv = float(gmv)
            keys.setdefault(keyword, [0,0,0,0,[]])
            keys[keyword][0] += show
            keys[keyword][1] += click
            keys[keyword][2] += cost
            keys[keyword][3] += gmv
            keys[keyword][4].append(value)
            cids.setdefault(cid, [0,0,0,0])
            cids[cid][0] += show
            cids[cid][1] += click
            cids[cid][2] += cost
            cids[cid][3] += gmv
    keys_sort = sorted(keys.items(), key = lambda x:x[1][1], reverse=True)
    for keyword, value in keys_sort:
        show = value[0]
        click = value[1]
        cost = value[2]
        gmv = value[3]
        key_num = len(value[4])
        values = value[4]
        values = [value.split('\t') for value in values]
        values_sort = sorted(values, key = lambda x:int(x[3]), reverse=True)
        esp = 0.00001
        key_ctr = click / (show + esp)
        for value in values_sort:
            show1 = int(value[2])
            click1 = int(value[3])
            ctr = click1 / (show1 + esp)
            cid = value[0]
            cid_post = cids.get(cid, [])
            cid_show = int(cid_post[0])
            cid_click = int(cid_post[1])
            cid_ctr = cid_click / (cid_show + esp)
            if click1 < 10 and ctr < 0.1: continue
            if ctr * 1.2 < key_ctr or ctr * 1.2 < cid_ctr: continue
            print(keyword + "\t" + "\t".join(value) + "\t" + str(show) + "\t" + str(click) +  \
                    "\t" + str(cost) + "\t" + str(gmv) + "\t" + "\t".join([str(i) for i in cid_post]) + "\t" + str(key_num))

def get_image(file_name, image_file, image_file_dir):
    cid_url = {}
    df = pd.read_csv(file_name, sep='\t')
    cids = df['component_id']
    urls = df['image_url']
    for cid, url in zip(cids, urls):
        cid_url[cid] = url
    out = open(image_file, 'w')
    out.write("component_id\timage_url\n")
    for cid, url in cid_url.items():
        out.write("\t".join([str(cid), url]) + "\n")
    out.close()

    os.environ['HTTP_PROXY'] = "http://9.21.0.122:11113"
    os.environ['HTTPS_PROXY'] = "http://9.21.0.122:11113"
    download(
        processes_count=8,
        thread_count=256,
        url_list=image_file,
        output_folder=image_file_dir,
        image_size=256,
        output_format="parquet",
        input_format="tsv",
        url_col="image_url",
        save_additional_columns=["component_id"],
        distributor="multiprocessing",
        retries=5,
    )

def process_image(images_dir, image_base64):
    image_files = os.listdir(images_dir)
    extension = '.parquet'
    image_files = [a for a in image_files if a.endswith(extension)]
    cids_base64 = {}
    out = open(image_base64, 'w')
    for image_file in image_files:
        image_file = os.path.join(images_dir, image_file)
        df = pd.read_parquet(image_file)
        cids = df['component_id']
        images = df["jpg"]
        statuss = df['status']
        for cid, image, status, in zip(cids, images, statuss):
            if status == "success":
                img = Image.open(BytesIO(image))
                img_buffer = BytesIO()
                img.save(img_buffer, format=img.format)
                byte_data = img_buffer.getvalue()
                base64_str = base64.b64encode(byte_data) # bytes
                base64_str = base64_str.decode("utf-8") # str
                out.write(str(cid) + "\t" + base64_str + "\n")
                cids_base64[cid] = base64_str
    out.close()
    print(len(cids_base64))
    return cids_base64

def get_sample(file_name, cids_base64, data_dir):
    train_image = os.path.join(data_dir, "train_imgs.tsv")
    train_pair = os.path.join(data_dir, "train_texts.jsonl")
    val_image = os.path.join(data_dir, "valid_imgs.tsv")
    val_pair = os.path.join(data_dir, "valid_texts.jsonl")
    test_image = os.path.join(data_dir, "test_imgs.tsv")
    test_pair = os.path.join(data_dir, "test_texts.jsonl")

    key_cids = {}
    df = pd.read_csv(file_name, sep='\t')
    keys = df['keyword']
    cids = df['component_id']
    clicks = df['click']
    all_clicks = df['all_click']
    for key, cid, click, all_click in zip(keys, cids, clicks, all_clicks):
        if cid not in cids_base64: continue
        click = int(click)
        all_click = int(all_click)
        all_num = math.pow(((math.log(all_click+4, 4))),2)
        sub_num = int(click / all_click * all_num +  math.log(click+4, 4))
        #print("\t".join([str(i) for i in [click, all_click, all_num, sub_num]]))
        key_cids.setdefault(key, [])
        for i in range(0, sub_num):
            key_cids[key].append(cid)
    key_num = len(key_cids)
    out1 = open(train_pair, 'w')
    out2 = open(val_pair, 'w')
    out3 = open(test_pair, 'w')
    cid1 = set()
    cid2 = set()
    cid3 = set()
    kid = 0
    for keyword in key_cids:
        cids = key_cids[keyword]
        kid += 1
        res = {}
        res["text_id"] = kid
        res["text"] = keyword
        train_cids = []
        val_cids = []
        test_cids = []
        for cid in cids:
            rd = random.random()
            if rd <= 0.95:
                train_cids.append(cid)
                cid1.add(cid)
            if rd > 0.95 and rd <= 1.0:
                val_cids.append(cid)
                cid2.add(cid)
            if rd > 1.0:
                test_cids.append(cid)
                cid3.add(cid)
        if len(train_cids) > 0:
            res["image_ids"] = train_cids
            out1.write(json.dumps(res, ensure_ascii=False) + "\n")
        if len(val_cids) > 0:
            res["image_ids"] = val_cids
            out2.write(json.dumps(res, ensure_ascii=False) + "\n")
        if len(test_cids) > 0:
            res["image_ids"] = test_cids
            out3.write(json.dumps(res, ensure_ascii=False) + "\n")

    out4 = open(train_image, 'w')
    out5 = open(val_image, 'w')
    out6 = open(test_image, 'w')
    for cid in cid1:
        base64 = cids_base64[cid]
        out4.write(str(cid) + "\t" + base64 + "\n")
    for cid in cid2:
        base64 = cids_base64[cid]
        out5.write(str(cid) + "\t" + base64 + "\n")
    for cid in cid3:
        base64 = cids_base64[cid]
        out6.write(str(cid) + "\t" + base64 + "\n")

if __name__ == '__main__':
    #data_sort()
    #exit()

    data_file = sys.argv[1]
    dataset_name = sys.argv[2]
    dataset_dir = "/apdcephfs_cq11/share_2973545/wenjieying/component/Chinese-CLIP/data/datasets"
    dataset_dir = os.path.join(dataset_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    image_file = os.path.join(dataset_dir, dataset_name + "_image_file") 
    image_file_dir = os.path.join(dataset_dir, dataset_name + "_image_file_dir")
    image_base64 = os.path.join(dataset_dir, dataset_name + "_image_base64")
    print(dataset_dir)
    print(image_file)
    print(image_file_dir)
    #get_image(data_file, image_file, image_file_dir)
    cids_base64 = process_image(image_file_dir, image_base64)
    #get_sample(data_file, cids_base64, dataset_dir)
    
