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

image_path = "/apdcephfs_cq3/share_2973545/image_generation/compoment/image/image_history_clip"

def get_df(df_files, tp):
    out_file = "image_cid_url.hdf5"
    time1 = time.time()
    if tp == "text":
        time2 = time.time()
        pds = [pd.read_parquet(df_file) for df_file in df_files]
        df = pd.concat(pds, ignore_index=True)
        time3 = time.time()
        print("read df" + str(time3 - time2))
        return df
    if tp == "image":
        #embs = numpy.zeros((11595676, 1024), dtype="float16")
        cids = []
        urls = []
        for i, df_file in zip(range(len(df_files)), df_files):
            time1 = time.time()
            df = pd.read_parquet(df_file, columns=['component_id', 'emb'])
            file_name = df_file.split('/')[-1]
            print(file_name)
            file_name = file_name.split("_")[1].split(".")[0]
            out_file = image_path + "/image_clip2_" + file_name + ".parquet"
            df.to_parquet(out_file)
            time2 = time.time()
            print(time2 - time1)
    time2 = time.time()
    print("read df" + str(time2 - time1))
    return df

if __name__ == '__main__':

    #1 get texts and images df, get key_cids images.hdf5
    images = ["/apdcephfs_cq3/share_2973545/wenjieying/component/data/images_vec/image_" + str(i).zfill(5) + ".parquet" for i in range(1160,1161)]
    df = get_df(images, "image")

    #2 from images.hdf5 select images_show to image_show.hdf5
    #get_h5_sub("../ann/image_cid_url.hdf5", "image_cid_url_show.hdf5", "../show_month_1107_all_cuid")
    #exit()
