#########################################################################
# File Name: demo.py
# Author: yingwenjie
# mail: yingwenjie@baidu.com
# Created Time: Tue 31 Oct 2023 11:24:44 AM CST
#########################################################################
import sys
sys.path.append("/apdcephfs_cq3/share_2973545/wenjieying/component/Chinese-CLIP")
import torch
from PIL import Image
from io import BytesIO
import numpy
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
import time

class ClipEmb(object):
    def __init__(self, model_path="/apdcephfs_cq11/share_2973545/wenjieying/component/Chinese-CLIP/clip_cn_vit-h-14.pt", model_name="ViT-H-14"):
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
                #image_input = self.preprocess(Image.open(image)).unsqueeze(0).to(self.device)
            except Exception as e:
                print("image_input is fail")
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
