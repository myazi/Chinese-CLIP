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
    key_dict = get_data_text(input_file)
    texts = list(key_dict.keys())
    texts_batch = split_list(texts, text_batch_size)
    embs_save = []
    texts_save = []
    for texts in texts_batch:
        embs = clipemb.get_text_emb(texts)
        for text in texts: texts_save.append(text)
        for emb in embs: embs_save.append(emb.tolist())
        step += 1
        if step % save_step == 0:
            embs_save_series = pd.Series(list(embs_save))
            text_embs_dict = {"key": texts_save, "emb": embs_save_series}
            text_df = pd.DataFrame(text_embs_dict)
            text_df.to_parquet(output_path + 'text_' + str(int(step/save_step)).zfill(5) + '.parquet')
            texts_save = []
            embs_save = []
    if len(embs_save) > 0:
        embs_save_series = pd.Series(list(embs_save))
        text_embs_dict = {"key": texts_save, "emb": embs_save_series}
        text_df = pd.DataFrame(text_embs_dict)
        text_df.to_parquet(output_path + 'text_' + str(int(step/save_step) + 1).zfill(5) + '.parquet')
    return text_df

def process_image_vec(clipemb, image_files, output_path):
    """
    predict imgae vector
    input image parquet file
    """
    image_batch_size = 1000
    file_index = 0
    for image_file in image_files:
        file_index += 1
        df = get_data_image(image_file)
        #df = df.head(100)
        images = df["jpg"]
        images_batch = split_list(images, image_batch_size)
        embs_save = []
        for images in images_batch:
            embs = clipemb.get_image_emb(images)
            for emb in embs: embs_save.append(emb.tolist())
        embs_save_series = pd.Series(embs_save)
        df["emb"] = embs_save_series
        df.to_parquet(output_path + 'image_' + str(file_index).zfill(5) + '.parquet')
    return df

if __name__ == '__main__':

    clipemb = ClipEmb()
    text_file = "/apdcephfs_cq3/share_2973545/wenjieying/component/data/601c5641-6efd-4161-becb-779793cf7021.txt_500W_2"
    image_files = ["/apdcephfs_cq3/share_2973545/wenjieying/component/images_sub/" + str(i).zfill(5) + ".parquet" for i in range(2)]
    output_path = "/apdcephfs_cq3/share_2973545/wenjieying/component/"
    output_path = "/apdcephfs_cq3/share_2973545/wenjieying/component/data/texts_vec_1107_show_1053_500_2/"

    #1 cal texts and images vector
    text_df = predict_text_vec(clipemb, text_file, output_path)
    #df = process_image_vec(clipemb, image_files, output_path)
