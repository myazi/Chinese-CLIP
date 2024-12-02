#########################################################################
# File Name: clip_score.py
# Author: yingwenjie
# mail: yingwenjie@tencent.com
# Created Time: Wed 03 Apr 2024 05:19:51 PM CST
#########################################################################
import sys
import os
import h5py
import numpy
import pandas as pd
from ClipEmb import ClipEmb 
#from img2dataset import download

def split_list(lst, batch_size):
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]
def process_text(clipemb, file_name):
    """
    predict text vector
    input: key cuids file
    """
    df = pd.read_csv(file_name, sep='\t')
    keys = df['keyword']
    keys = set(keys)
    texts = list(keys)
    text_batch_size = 1000
    texts_batch = split_list(texts, text_batch_size)
    embs_save = []
    for texts in texts_batch:
        embs = clipemb.get_text_emb(texts)
        for emb in embs: embs_save.append(emb)
    key_embs = {}
    for text, embs in zip(texts, embs_save):
        key_embs[text] = embs
    return key_embs

def get_image(file_name, ann_file, image_file, image_file_dir):
    cid_url = {}
    df1 = pd.read_csv(file_name, sep='\t')
    df2 = pd.read_csv(ann_file, sep='\t')
    df = pd.concat([df1, df2])
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
        retries=3,
        number_sample_per_shard=100000000,
    )

def process_image(clipemb, image_file_dir):
    image_files = os.listdir(image_file_dir)
    extension = '.parquet'
    image_files = [a for a in image_files if a.endswith(extension)]
    df = pd.read_parquet(os.path.join(image_file_dir, image_files[0]))
    cids = df['component_id']
    images = df["jpg"]

    image_batch_size = 100
    images_batch = split_list(images, image_batch_size)
    embs_save = []
    for images in images_batch[0:1]:
        embs = clipemb.get_image_emb(images)
        for emb in embs: embs_save.append(emb:
    cid_embs = {}
    for cid, embs in zip(cids, embs_save):
        cid_embs[cid] = embs
    return cid_embs

def get_sim(file_name, ann_file, key_embs, cid_embs, out_file):
    df1 = pd.read_csv(file_name, sep='\t')
    df2 = pd.read_csv(ann_file, sep='\t')
    df1['tag'] = df1.apply(lambda row: row['is_good_image'] - row['is_bad_image'], axis=1)
    df2['tag'] = "ann"
    df = pd.concat([df1, df2])
    keys = df['keyword']
    cids = df['component_id']
    urls = df['image_url']
    tags = df['tag']
    out = open(out_file, 'w')
    out.write("keyword\tcomponent_id\timage_url\ttag\tsim\n")
    for keyword, cid, url, tag in zip(keys, cids, urls, tags):
        key_emb = key_embs[keyword]
        cid_emb = cid_embs[cid]
        cosine_per_image = numpy.dot(key_emb, numpy.transpose(cid_emb))
        out.write(keyword + "\t" + str(cid) + "\t" + url + "\t" + str(tag) + "\t" + str(cosine_per_image) + "\n")

if __name__ == '__main__':
    data_file = sys.argv[1]
    dataset_name = sys.argv[2]
    model_name = sys.argv[3]
    dataset_dir = "/apdcephfs_cq11/share_2973545/wenjieying/component/Chinese-CLIP/data/test"
    dataset_dir = os.path.join(dataset_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    ann_file = os.path.join(dataset_dir, dataset_name + "_ann_res")
    image_file = os.path.join(dataset_dir, dataset_name + "_image_file") 
    image_file_dir = os.path.join(dataset_dir, dataset_name + "_image_file_dir")
    res_file = os.path.join(dataset_dir, dataset_name + "_score_" + model_name)

    #get_image(data_file, ann_file, image_file, image_file_dir)
    #exit()
    clipemb = ClipEmb()
    key_embs = process_text(clipemb, data_file)
    #get_image(data_file, ann_file, image_file, image_file_dir)
    cid_embs = process_image(clipemb, image_file_dir)
    get_sim(data_file, ann_file, key_embs, cid_embs, res_file)

