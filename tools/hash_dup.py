#########################################################################
# File Name: hash_dup.py
# Author: yingwenjie
# mail: yingwenjie@baidu.com
# Created Time: Wed 15 Nov 2023 04:40:51 PM CST
#########################################################################
import sys
import time
from typing import Callable, Dict
import numpy as np
import multiprocessing

# 暴力搜索
class BruteForce:
    """
    Class to perform search using a Brute force.
    """
    def __init__(self, hash_dict: Dict, distance_function: Callable) -> None:
        """
        Initialize a dictionary for mapping file names and corresponding hashes and a distance function to be used for
        getting distance between two hash strings.

        Args:
            hash_dict: Dictionary mapping file names to corresponding hash strings {filename: hash}
            distance_function:  A function for calculating distance between the hashes.
        """
        self.distance_function = distance_function
        self.hash_dict = hash_dict  # database

    def search1(self, query: str, tol: int) -> Dict[str, int]:
        """
        Function for searching using brute force.

        Args:
            query: hash string for which brute force needs to work.
            tol: distance upto which duplicate is valid.

        Returns:
            List of tuples of the form [(valid_retrieval_filename1: distance), (valid_retrieval_filename2: distance)]
        """
        return [
            (item, self.distance_function(query, self.hash_dict[item]))
            for item in self.hash_dict
            if self.distance_function(query, self.hash_dict[item]) <= tol
        ]

    def search2(self, query: str, tol: int, idx_list: list) -> Dict[str, int]:
        """
        Function for searching using brute force.

        Args:
            query: hash string for which brute force needs to work.
            tol: distance upto which duplicate is valid.

        Returns:
            List of tuples of the form [(valid_retrieval_filename1: distance), (valid_retrieval_filename2: distance)]
        """
        return [ item for item in idx_list 
                if self.distance_function(query, self.hash_dict[item]) <= tol
        ]

def hamming_distance(hash1: str, hash2: str) -> float:
    """
    Calculate the hamming distance between two hashes. If length of hashes is not 64 bits, then pads the length
    to be 64 for each hash and then calculates the hamming distance.

    Args:
        hash1: hash string
        hash2: hash string

    Returns:
        hamming_distance: Hamming distance between the two hashes.
    """
    #hash1_bin = bin(int(hash1, 16))[2:].zfill(
    #    64
    #)  # zfill ensures that len of hash is 64 and pads MSB if it is < A
    #hash2_bin = bin(int(hash2, 16))[2:].zfill(64)
    #return np.sum([i != j for i, j in zip(hash1_bin, hash2_bin)])
    hash1 = int(hash1, 16)
    hash2 = int(hash2, 16)
    dist = bin(hash1 ^ hash2).count('1')
    return dist

def hash_dup1(hashs, thd=15, topk=100):
    
    rank_to_hash = dict(zip(range(len(hashs)), hashs))
    brute_force = BruteForce(rank_to_hash, hamming_distance)

    result_list = [True] * len(hashs)
    time_all = 0
    for idx in range(0, len(hashs)):
        if result_list[idx] == False: continue
        # 暴力搜索结果
        source_hash = hashs[idx]
        time1 = time.time()
        search_results = brute_force.search1(source_hash, thd)
        time2 = time.time()
        time_all += time2 - time1
        if len(search_results) <= 1: continue

        for i, dis in search_results:
            # 将当前图片之后的重复图片删除
            if i > idx:
                result_list[i] = False
            # 如果当前图片之前有重复图片且未被删除，将自身删除
            elif i < idx and result_list[i]:
                result_list[idx] = False
    print("hash:" + str(time_all))
    idx_list = [i for i in range(len(result_list)) if result_list[i] == True]
    return idx_list[0:topk]

def hash_dup2(hashs, thd=15, topk=100):
    
    rank_to_hash = dict(zip(range(len(hashs)), hashs))
    brute_force = BruteForce(rank_to_hash, hamming_distance)

    idx_list = []
    time_all = 0
    for idx in range(0, len(hashs)):
        # 暴力搜索结果
        source_hash = hashs[idx]
        time1 = time.time()
        search_results = brute_force.search2(source_hash, thd, idx_list)
        time2 = time.time()
        time_all += time2 - time1
        if len(search_results) <= 0: idx_list.append(idx)
        if len(idx_list) >= topk: break

    return idx_list

def get_cid2hash(file_names):
    cid_hash = {}
    for file_name in file_names:
        with open(file_name) as f:
            for line in f:
                line_list = line.strip('\n').split('\t')
                if len(line_list) < 2: continue
                cid, hash_value = line_list[0:2]
                cid_hash[cid] = hash_value
    return cid_hash

def process_file(ann_file):
    start_time = time.time()
    key_dict = {}
    with open(ann_file) as f:
        for line in f:
            line_list = line.strip('\n').split('\t')
            if len(line_list) < 3: continue
            key, cid, score = line_list[0:3]
            key_dict.setdefault(key, [])
            hash_value = cid2hash.get(cid, "0")
            if hash_value == "0":
                print(cid + ": " + "phash code is error")
            key_dict[key].append([cid, score, hash_value])
    ann_file_dup = ann_file + "_dup"
    f_out = open(ann_file_dup, 'w')
    for key in key_dict:
        one_res = key_dict[key]
        hashs = [i[2] for i in one_res]
        time1 = time.time()
        idx_list = hash_dup2(hashs)
        time2 = time.time()
        print("all:" + str(time2 - time1))
        print("\t".join([str(i) for i in idx_list]))
        for idx in idx_list:
            f_out.write(key + "\t" + "\t".join(one_res[idx][0:2]) + "\n")
    end_time = time.time()
    use_time = end_time - start_time
    print("one_file use time:" + "\t" + str(use_time))

if __name__ == '__main__':
    global cid2hash
    hash_file_names = ["/apdcephfs_cq3/share_2973545/wenjieying/component/data/phash/" + str(i).zfill(5) + ".csv" for i in range(0,1160)]
    cid2hash = get_cid2hash(hash_file_names)
    ann_file_names = ["/apdcephfs_cq3/share_2973545/wenjieying/component/data/texts_vec_1127_show/text_" + str(i).zfill(5) + "_ann" for i in range(1,1073)]
    #ann_file_names = ["review_data"]

    start_time = time.time()
    with multiprocessing.Pool(processes=20) as pool:
        results = pool.map(process_file, ann_file_names)
    end_time = time.time()
    print("test all time: " + str(end_time - start_time))
