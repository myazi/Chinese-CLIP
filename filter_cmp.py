#########################################################################
# File Name: filter_cmp.py
# Author: yingwenjie
# mail: yingwenjie@baidu.com
# Created Time: Fri 17 Nov 2023 10:18:29 AM CST
#########################################################################
import sys
import time
from collections import defaultdict
import multiprocessing

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_w = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, w):

        current = self.root
        for c in w:
            current = current.children[c]

        current.is_w = True

    def get_lexicon(self, sentence):
        result = set()
        for i in range(len(sentence)):
            #if len(result) != 0: break
            current = self.root
            for j in range(i, len(sentence)):
                current = current.children.get(sentence[j])
                if current is None:
                    break

                if current.is_w:
                    result.add(sentence[i:j + 1])

        return result

def get_brand_word(file_name):
    trie = Trie()
    filter_word = set()
    with open("/apdcephfs_cq3/share_2973545/wenjieying/component/data/texts_vec_1107_show_1053_500_ann/brand_num_filter") as f:
        for line in f:
            word = line.strip('\n').split('\t')[0]
            filter_word.add(word)
    with open(file_name) as f:
        for line in f:
            word = line.strip('\n')
            if word in filter_word: continue
            trie.insert(word)
    print("brand done")
    return trie

def filter_cmp1(file_name):
    ann_file_brand = file_name + "_filter_brand1"
    f_out = open(ann_file_brand, 'w')
    with open(file_name) as f:
        for line in f:
            flag = 1
            line_list = line.strip('\n').split('\t')
            if len(line_list) < 3: continue
            key, cid, score = line_list[0:3]
            key_res = brand_trie.get_lexicon(key)
            word = cid_info.get(cid, "")
            word, url = word.split('\t')
            word_res = brand_trie.get_lexicon(word)
            if len(word_res) != 0:
                for key in word_res:
                    if key not in key_res:
                        f_out.write(line.strip('\n') + "\t" + url + "\t" + word + "\t" + key + "\n")
                        flag = 0
                        break
            if flag:
                f_out.write(line.strip('\n') + "\t" + url + "\t" + word + "\t" + "none" + "\n")

def get_cid_info(file_name):
    cid_info = {}
    with open(file_name) as f:
        for line in f:
            line_list = line.strip('\n').split('\t')
            if(len(line_list) < 4): continue
            cid = line_list[0]
            url = line_list[1]
            word = line_list[3]
            cid_info[cid] = word + "\t" + url
    print("cid_info done")
    return cid_info

def get_text_brand(file_names):
    key_ners = {}
    for file_name in file_names:
        with open(file_name) as f:
            for line in f:
                ners = []
                key, labels = line.strip('\n').split('\t')
                label_list = labels.split(':::')
                if len(label_list) <= 0: continue
                for label in label_list:
                    labels = label.split('==')
                    if len(labels) != 4:
                        #print(key)
                        continue
                    ner, tp, score, tp2 = labels
                    if float(score) < 0.5: continue
                    if tp == "BRAND" or tp == "SOFTWARE" or tp == "INSITE" or "GAME" in tp:
                        ners.append(ner)
                key_ners[key] = ners
    print("ners done")
    return key_ners

def get_ocr_brand(file_name):
    cid_ocr = {}
    cid_ocr_ners = {}
    with open(file_name) as f:
        for line in f:
            ners = []
            cid, key, labels = line.strip('\n').split('\t')
            cid_ocr[cid] = key
            label_list = labels.split(':::')
            if len(label_list) <= 0: continue
            for label in label_list:
                labels = label.split('==')
                if len(labels) != 4:
                    #print(key)
                    continue
                ner, tp, score, tp2 = labels
                if float(score) < 0.5: continue
                if tp == "BRAND" or tp == "SOFTWARE" or tp == "INSITE" or "GAME" in tp:
                    ners.append(ner)
            cid_ocr_ners[cid] = ners
    print("cid ocr ners done")
    return cid_ocr, cid_ocr_ners

def filter_cmp2(file_name):
    ann_file_brand = file_name + "_filter_brand_ocr"
    f_out = open(ann_file_brand, 'w')
    with open(file_name) as f:
        for line in f:
            flag = 1
            line_list = line.strip('\n').split('\t')
            if len(line_list) < 3: continue
            key, cid, score = line_list[0:3]
            key_res = key_brand.get(key, [])
            ocr = cid_ocr.get(cid, "no")
            if 'www' in ocr or 'blog' in ocr or 'api' in ocr or '.com' in ocr \
                    or '.org' in ocr or '.net' in ocr or '.edu' in ocr \
                    or '.gov' in ocr or '.co' in ocr or '.io' in ocr \
                    or '.info' in ocr:
                continue
            ocr_res = cid_ocr_brand.get(cid, [])
            word = cid_info.get(cid, "")
            word, url = word.split('\t')
            word_res = key_brand.get(word, [])
            if len(ocr) > 50:
                if len(word_res) != 0:
                    for key in word_res:
                        if key not in key_res:
                            #f_out.write(line.strip('\n') + "\t" + url + "\t" + word + "\t" + ocr + "\t" + key + "\n")
                            #f_out.write(line.strip('\n') + "\n")
                            flag = 0
                            break
            else:
                if len(ocr_res) != 0:
                    for key in ocr_res:
                        if key not in key_res:
                            #f_out.write(line.strip('\n') + "\t" + url + "\t" + word + "\t" + ocr + "\t" + key + "\n")
                            #f_out.write(line.strip('\n') + "\n")
                            flag = 0
                            break
            if flag:
                #f_out.write(line.strip('\n') + "\t" + url + "\t" + word + "\t" + ocr + "\t" + "none" + "\n")
                f_out.write(line.strip('\n') + "\n")

if __name__ == '__main__':
    #brand_file_name = "/apdcephfs_cq3/share_2973545/wenjieying/component/data/all_brand_total_20230907.txt"
    cid_file_name = "/apdcephfs_cq3/share_2973545/wenjieying/component/data/41370a5f-1cbb-4921-9896-99ad986fdb90.txt"
    brand_file_names = ["/apdcephfs_cq3/share_2973545/wenjieying/component/data/keyword_label_brand/part_" + str(i) + "_ner" for i in range(0,30)]
    cid_ocr_file_name = "/apdcephfs_cq3/share_2973545/wenjieying/component/data/ocr_brand/cid_ocr_all_csv_ner" 
    ann_file_names = ["/apdcephfs_cq3/share_2973545/wenjieying/component/data/texts_vec_1127_show/text_" + str(i).zfill(5) + "_ann_dup" for i in range(1,1073)]
    #global brand_trie
    global cid_info
    global key_brand
    global cid_ocr_brand
    global cid_ocr
    #brand_trie = get_brand_word(brand_file_name)
    cid_info = get_cid_info(cid_file_name)
    key_brand = get_text_brand(brand_file_names)
    cid_ocr, cid_ocr_brand = get_ocr_brand(cid_ocr_file_name)

    start_time = time.time()
    with multiprocessing.Pool(processes=20) as pool:
        #results = pool.map(filter_cmp1, ann_file_names)
        results = pool.map(filter_cmp2, ann_file_names)
    end_time = time.time()
    print("test all time: " + str(end_time - start_time))
