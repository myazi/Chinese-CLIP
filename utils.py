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
import h5py

def split_list(lst, batch_size):
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]

def get_data_text(text_file):
    """
    from show_file read key and cuids
    """
    time1 = time.time()
    key_dict = {}
    with open(text_file) as f:
        for line in f:
            line_list = line.strip('\n').split("\t")
            key, cid = line_list[0:2]
            key_dict.setdefault(key, set())
            key_dict[key].add(cid)
    time2 = time.time()
    print("load text" + str(time2 - time1))
    return key_dict

def get_data_image(image_file):
    """
    from img2data read parquet
    """
    time2 = time.time()
    df = pd.read_parquet(image_file)
    time3 = time.time()
    print("load image" + str(time3 - time2))
    return df
