#########################################################################
# File Name: demo.py
# Author: yingwenjie
# mail: yingwenjie@baidu.com
# Created Time: Tue 31 Oct 2023 11:24:44 AM CST
#########################################################################
import sys
import torch
from PIL import Image
from io import BytesIO
import numpy
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
import pandas as pd
import time
import h5py

class ClipEmb(object):
    def __init__(self, model_path="./", model_name="ViT-H-14"):
        self.model_path = model_path
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = load_from_name(self.model_name, device=self.device, download_root=self.model_path)
        self.model.eval()

    def get_image_emb(self, images):
        images_input = []
        for image in images:
            try:
                image_input = self.preprocess(Image.open(BytesIO(image))).unsqueeze(0).to(self.device)
            except Exception as e:
                image_input = torch.zeros((1, 3, 224, 224), device=self.device)
            images_input.append(image_input)
        with torch.no_grad():
            images_inputs = torch.cat(images_input, dim=0)
            images_features = self.model.encode_image(images_inputs)
            #images_features /= images_features.norm(dim=-1, keepdim=True)
            images_features = images_features.cpu().numpy() #todo gpu norm
            images_norms = numpy.linalg.norm(images_features, axis=1)
            images_features /= images_norms[:, numpy.newaxis]
        return images_features

    def get_text_emb(self, texts):
        with torch.no_grad():
            texts = clip.tokenize(texts).to(self.device)
            texts_features = self.model.encode_text(texts)
            texts_features = texts_features.cpu().numpy()
            texts_norms = numpy.linalg.norm(texts_features, axis=1)
            texts_features /= texts_norms[:, numpy.newaxis]
        return texts_features
    
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

def process_text_vec(clipemb, input_file, output_path):
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

def get_sim(text_df, image_df, key_dict, offset):
    keys = text_df["key"]
    texts_features = numpy.vstack(text_df["emb"].to_numpy())
    print(texts_features.shape)
    cids = image_df["component_id"]
    images_features = numpy.vstack(image_df["emb"].to_numpy())
    print(images_features.shape)
    index2cid = []
    cid2index = {}
    for i in range(len(cids)):
        index2cid.append(str(cids[i]))
        cid2index[str(cids[i])] = i

    line = 0
    for key in keys:
        print(key + "@" + str(line) + "\t" + str(key_dict[key]))
        cids = []
        cid_indexs = [] 
        for cid in key_dict[key]:
            index = cid2index.get(cid, -1)
            if index == -1:
                print("miss_cid" + "\t" + cid)
                continue
            cids.append(cid)
            cid_indexs.append(index)
        cid_embs = images_features[cid_indexs]
        cosine_per_image = numpy.dot(texts_features[[line]], numpy.transpose(cid_embs))
        
        print(cosine_per_image.shape)
        n, m = cosine_per_image.shape
        for i in range(n):
            for j in range(m):
                print(key + "\t" + cids[j] + "\t" + str(cosine_per_image[i,j]))
        line += 1

    """
    # cosine similarity
    cosine_per_image = numpy.dot(texts_features, numpy.transpose(images_features))
    print(cosine_per_image.shape)
    line = 0
    for key in keys:
        print(key + "@" + str(line) + "\t" + str(key_dict[key]))
        for cid in key_dict[key]:
            index = cid2index.get(cid, -1)
            if index == -1:
                print("miss_cid" + "\t" + cid)
                continue
            print(key + "\t" + cid + "\t" + str(cosine_per_image[line][index]))
        line += 1

    n, m = cosine_per_image.shape
    for i in range(n):
        for j in range(m):
            if(cosine_per_image[i,j] > 0.3):
                print(keys[i + offset * 10000] + "\t" + index2cid[j] + "\t" + str(cosine_per_image[i,j]))
    """

def get_sim_vec(keys, texts_features, cids, images_features, key_dict):

    index2cid = []
    cid2index = {}
    for i in range(len(cids)):
        index2cid.append(str(cids[i].decode('utf-8')))
        cid2index[str(cids[i].decode('utf-8'))] = i

    line = 0
    for key in keys:
        print(key + "@" + str(line) + "\t" + str(key_dict[key]))
        cids = []
        cid_indexs = [] 
        for cid in key_dict[key]:
            index = cid2index.get(cid, -1)
            if index == -1:
                print("miss_cid" + "\t" + cid)
                continue
            cids.append(cid)
            cid_indexs.append(index)
        cid_indexs = sorted(cid_indexs)
        cid_embs = images_features[cid_indexs]
        cosine_per_image = numpy.dot(texts_features[[line]], numpy.transpose(cid_embs))
        
        print(cosine_per_image.shape)
        n, m = cosine_per_image.shape
        for i in range(n):
            for j in range(m):
                print(key + "\t" + cids[j] + "\t" + str(cosine_per_image[i,j]))
        line += 1

if __name__ == '__main__':

    clipemb = ClipEmb("./clip_cn_vit-h-14.pt")
    text_file = "/apdcephfs_cq11/share_2973545/image_generation/component/keyword/a"
    image_files = ["./data/dataset/b/b_image_file_dir" + str(i).zfill(5) + ".parquet" for i in range(1)]

    #1 cal texts and images vector
    text_df = process_text_vec(clipemb, text_file, output_path)
    image_df = process_image_vec(clipemb, image_files, output_path)

    get_sim_vec(keys, texts_embs, cids, images_embs, key_dict)

    #2 get texts and images vector, get key_cids
    """
    key_dict = get_data_text(text_file)
    df_texts = [output_path + "text_" + str(i).zfill(5) + ".parquet" for i in range(1,25)]
    df_images = [output_path + "image_" + str(i).zfill(5) + ".parquet" for i in range(1,1160)]
    print(texts)
    print(images)
    pds = [pd.read_parquet(df_file) for df_file in df_texts]
    text_df = pd.concat(pds, ignore_index=True)

    time1 = time.time()
    for i in range(24):
        text_df_sub = text_df.iloc[10000 * i: min((i+1) * 10000,text_df.shape[0])]
        get_sim(text_df_sub, df, key_dict, i)
    #get_sim(text_df, df, key_dict, 0)
    time2 = time.time()
    print(str(time2 - time1))
    """

    #3. get texts and image vector, get key_cids
    key_dict = get_data_text(text_file)
    df_texts = [output_path + "text_" + str(i).zfill(5) + ".parquet" for i in range(1,2)]
    pds = [pd.read_parquet(df_file) for df_file in df_texts]
    text_df = pd.concat(pds, ignore_index=True)

    keys = text_df["key"]
    texts_embs = numpy.vstack(text_df["emb"].to_numpy())
    hdf5_f = h5py.File(output_path + "ann/image_cid_url.hdf5", 'r')
    images_embs = hdf5_f["vector"]
    cids = hdf5_f["cid"]
    urls = hdf5_f["url"]
    get_sim_vec(keys, texts_embs, cids, images_embs, key_dict)
