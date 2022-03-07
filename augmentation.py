import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util
import torch.utils.data as data
import itertools

from args import get_train_args
from collections import OrderedDict
from json import dumps, dump
from models import CoAttention
# from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from util import collate_fn, SQuAD

# TODO: should we do this?
# import nltk
# nltk.download('wordnet')
# from nltk.corpus import wordnet 

# stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
# 			'ours', 'ourselves', 'you', 'your', 'yours', 
# 			'yourself', 'yourselves', 'he', 'him', 'his', 
# 			'himself', 'she', 'her', 'hers', 'herself', 
# 			'it', 'its', 'itself', 'they', 'them', 'their', 
# 			'theirs', 'themselves', 'what', 'which', 'who', 
# 			'whom', 'this', 'that', 'these', 'those', 'am', 
# 			'is', 'are', 'was', 'were', 'be', 'been', 'being', 
# 			'have', 'has', 'had', 'having', 'do', 'does', 'did',
# 			'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
# 			'because', 'as', 'until', 'while', 'of', 'at', 
# 			'by', 'for', 'with', 'about', 'against', 'between',
# 			'into', 'through', 'during', 'before', 'after', 
# 			'above', 'below', 'to', 'from', 'up', 'down', 'in',
# 			'out', 'on', 'off', 'over', 'under', 'again', 
# 			'further', 'then', 'once', 'here', 'there', 'when', 
# 			'where', 'why', 'how', 'all', 'any', 'both', 'each', 
# 			'few', 'more', 'most', 'other', 'some', 'such', 'no', 
# 			'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
# 			'very', 's', 't', 'can', 'will', 'just', 'don', 
# 			'should', 'now', '']
def main(args):
    # train_dataset = SQuAD(args.train_record_file, args.use_squad_v2)
    # print(train_dataset.context_idxs)
    alpha = 10
    with open('./data/train-v2.0.json', 'r') as original:
        train_dict = json_load(original)
    # print(len(train_dict['data']))
    augmented = train_dict.copy()
    for example in augmented['data']:
        aug_example = example.copy()
        for bundle in aug_example['paragraphs']:
            # print(bundle['qas'][0])
            sentences = bundle['context'].split('.')
            for sentence in sentences:
                if random.choice([True, False]):
                    n = len(sentence) // alpha
                    swap(sentence, n)
                else:
                    answers = list(itertools.chain([entry['answers'].split(' ') for entry in bundle['qas']]))
                    p = 1/alpha
                    delete(sentence, answers, p)
        augmented['data'].append(aug_example)
    with open('augmented_train.json', 'w') as f:
        dump(augmented, f)
        
        
    

def swap(sentence, n):
    sentence = sentence.split(' ')
    new = sentence.copy()
    for _ in range(n):
        idx1 = random.randint(0, len(sentence)-1)
        idx2 = random.randint(idx1 + 1, len(sentence)-1)
        new[idx1], new[idx2] = new[idx2], new[idx1]
    new = new.join(' ')
    return new

def delete(sentence, answers, p):
    sentence = sentence.split(' ')
    if len(sentence) == 1: return sentence
    new = []
    for word in sentence:
        if random.random() > p or word in answers:
            new.append(word)
    if len(new) == 0: return sentence[0]
    new = new.join(' ')
    return new

if __name__ == '__main__':
    main(get_train_args())
