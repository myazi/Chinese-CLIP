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
from ClipEmb import ClipEmb 
from utils import split_list, get_data_text, get_data_image

def predict_text_vec(clipemb, input_file, output_path):
    """
    predict text vector
    input: key cuids file
    """
    text_batch_size = 1000
    step = 0
    save_step = 10
    texts = get_data_text(input_file)
    texts_batch = split_list(texts, text_batch_size)
    embs_save = []
    texts_save = []
    for texts in texts_batch:
        embs = clipemb.get_text_emb(texts)
        for text in texts: texts_save.append(text)
        for emb in embs: embs_save.append(emb.tolist())
        if step % save_step == 0:
            embs_save_series = pd.Series(list(embs_save))
            text_embs_dict = {"key": texts_save, "emb": embs_save_series}
            text_df = pd.DataFrame(text_embs_dict)
            text_df.to_parquet(output_path + 'text_' + str(int(step/save_step)).zfill(5) + '.parquet')
            texts_save = []
            embs_save = []
        step += 1
    if len(embs_save) > 0:
        embs_save_series = pd.Series(list(embs_save))
        text_embs_dict = {"key": texts_save, "emb": embs_save_series}
        text_df = pd.DataFrame(text_embs_dict)
        text_df.to_parquet(output_path + 'text_' + str(int(step/save_step) + 1).zfill(5) + '.parquet')
    return text_df

def process_image_vec(clipemb, image_files, output_path, file_index=0):
    """
    predict imgae vector
    input image parquet file
    """
    image_batch_size = 1000
    for image_file in image_files:
        df = get_data_image(image_file)
        #df = df.head(100)
        images = df["jpg"]
        cids = df['component_id']
        images_batch = split_list(images, image_batch_size)
        embs_save = []
        for images in images_batch:
            embs = clipemb.get_image_emb(images)
            #for emb in embs: embs_save.append(emb.tolist())
            for cid, emb in zip(cids, embs):
                #print(str(cid) + "\t" + " ".join([str(i) for i in emb]))
                embs_save.append(emb.tolist())
        embs_save_series = pd.Series(embs_save)
        df["emb"] = embs_save_series
        df.to_parquet(output_path + 'image_clip_epoch_10_' + str(file_index).zfill(5) + '.parquet')
        file_index += 1
    return df

def get_keyword_image_score(clipemb, image_file):
    keyword_embs = clipemb.get_text_emb(["直播 视频"])
    df = get_data_image(image_file)
    images = df["jpg"]
    image_embs = clipemb.get_image_emb(images)
    print(keyword_embs)
    print(image_embs)
    print(numpy.dot(keyword_embs, image_embs.T))

if __name__ == '__main__':
    
    model_path = "/apdcephfs_cq11/share_2973545/wenjieying/component/Chinese-CLIP/data/experiments/0125_0611_sort_clip_cn_vit-h-14/checkpoints/epoch10.pt"
    clipemb = ClipEmb(model_path=model_path)

    #1 cal texts vector
    text_file = "./0418_test.txt_keyword"
    output_path = "./0418_test_10"
    text_df = predict_text_vec(clipemb, text_file, output_path)

    #2 cal images vector
    image_files = ["/apdcephfs_cq11/share_2973545/wenjieying/component/Chinese-CLIP/data/datasets/cid_url_0230_0617/cid_url_0230_0617_image_file_dir/" + str(i).zfill(5) + ".parquet" for i in range(0, 282)]
    output_path = "/apdcephfs_cq11/share_2973545/wenjieying/component/Chinese-CLIP/data/datasets/cid_url_0230_0617/cid_url_0230_0617_image_file_dir/"
    df = process_image_vec(clipemb, image_files, output_path, 0)


    #text_file = "/apdcephfs_cq11/share_2973545/wenjieying/component/Chinese-CLIP/0422_all_keyword"
    #output_path = "/apdcephfs_cq11/share_2973545/wenjieying/component/Chinese-CLIP/0422_all_keyword_clip/"
    #text_df = predict_text_vec(clipemb, text_file, output_path)
