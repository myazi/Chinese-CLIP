#########################################################################
# File Name: parquet_read.py
# Author: yingwenjie
# mail: yingwenjie@tencent.com
# Created Time: Tue 11 Jun 2024 08:41:59 PM CST
#########################################################################
import sys
import os
import pandas as pd
import h5py
import numpy

def write_parquet(path, all_files_names, cur_clip_path):
    all_data = pd.DataFrame()
    file_counter = 1
    cid_uniq = set()
    for file_name in all_file_names:
        print(file_name)
        df = pd.read_parquet(os.path.join(path, file_name))
        components = df['component_id'].astype(str)
        indexs = []
        for i, cid in enumerate(components):
            if cid in cid_uniq:
                continue
            else:
                cid_uniq.add(cid)
            indexs.append(i)
        df = df.iloc[indexs]
        all_data = pd.concat([all_data, df])
        while len(all_data) >= 10000:
            chunk_df = all_data[:10000]
            output_file_name = "image_clip_" + str(file_counter).zfill(5) + ".parquet"
            output_file_path = os.path.join(cur_clip_path, output_file_name)
            chunk_df.to_parquet(output_file_path)
            print(f"Saved chunk {file_counter} to {output_file_path}")
            # 更新 all_data 和文件计数器
            all_data = all_data[10000:]
            file_counter += 1

    # 保存剩余的数据
    if not all_data.empty:
        output_file_name = "image_clip_" + str(file_counter).zfill(5) + ".parquet"
        output_file_path = os.path.join(cur_clip_path, output_file_name)
        all_data.to_parquet(output_file_path)
        print(f"Saved chunk {file_counter} to {output_file_path}")

cid2url = {}
def get_url():
    i = 0
    with open("/apdcephfs_cq11/share_2973545/image_generation/component/image/image_history_info") as f:
        for line in f:
            line_list = line.strip('\n').split('\t')
            if len(line_list) < 3: continue
            cid, url, label = line_list[0:3]
            cid2url[cid] = url
            #if i > 100000: break
            #i+=1
    print("cid2url done")

def parquet_read(path):
    cid_set = set()
    with open("0128_0628_show") as f:
        for line in f:
            cid = line.strip('\n').split('\t')[0]
            cid_set.add(cid)
    print(len(cid_set))

    cid_uniq = set()
    embs_h5 = numpy.zeros((3416825, 1024), dtype="float16")
    cids_h5 = []
    urls_h5 = []
    i = 0

    for image in all_file_names:
        #if "image_clip_epoch_10_" not in image: continue
        print(image)
        df = pd.read_parquet(path + "/" + image, columns=['component_id', 'emb'])
        cids = df["component_id"].astype(str)
        embs = df["emb"]
        for cid, emb, in zip(cids, embs):
            if cid in cid_set and cid not in cid_uniq:
                embs_h5[i:i+1, :] = numpy.asarray(emb, dtype=numpy.float16)
                cids_h5.append(cid)
                url = cid2url.get(str(cid), "")
                urls_h5.append(url)
                i += 1
                #print(cid + "\t" + " ".join([str(i) for i in emb]))
                cid_uniq.add(cid)
    print(len(cids_h5))
    print(embs_h5.shape)
    out_file = "0128_0628_show_url.hdf5"
    f = h5py.File(out_file, 'w')
    f.create_dataset("vector", data=embs_h5)
    f.create_dataset("cid", data=cids_h5)
    f.create_dataset("url", data=urls_h5)
    f.close()

if __name__ == '__main__':
    path = sys.argv[1]
    all_file_names = os.listdir(path)
    output = "./image_history_clip_uniq"
    #write_parquet(path, all_file_names, output)
    #get_url()
    parquet_read(path)
    
