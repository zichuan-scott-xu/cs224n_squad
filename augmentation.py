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
import copy
import string

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
    augmented = copy.deepcopy(train_dict)
    print(len(train_dict['data']), ' total')
    for example in train_dict['data']:
        aug_example = copy.deepcopy(example)
        for bundle in aug_example['paragraphs']:
            # print(bundle['qas'][0])
            # print(bundle['context'])
            sentences = bundle['context'].split('.')
            new_context = []
            answers_list = [entry['answers'][0]['text'].split(' ') for entry in bundle['qas'] if len(entry['answers']) > 0]
            answers = list(itertools.chain(*answers_list))
            # print(answers)
            for sentence in sentences:
                
                if random.choice([True, False]):
                    n = len(sentence) // alpha
                    new_context.append(swap(sentence, answers, n))
                else:
                    # print(bundle['qas'][0]['answers'][0].keys())
                    p = 1/alpha
                    new_context.append(delete(sentence, answers, p))
            new_context_str = ' '.join((sen + '.') for sen in new_context)
            new_context_str = ' '.join(new_context_str.split())
            new_context_str = new_context_str.strip(string.punctuation).strip()
            bundle['context'] = new_context_str
            for entry in bundle['qas']:
                if len(entry['answers']) > 0:
                    pair = entry['answers'][0]
                    pair['answer_start'] = new_context_str.find(pair['text'])
                    if pair['answer_start'] == -1:
                        entry['answers'] = []
                    # print(new_context_str)
                    # print(pair['answer_start'])
                    # print(new_context_str[pair['answer_start']:pair['answer_start'] + len(pair['text'])])
                    # print(pair['text'])
                    # assert(new_context_str[pair['answer_start']:pair['answer_start'] + len(pair['text'])] == pair['text'])
        augmented['data'].append(aug_example)
        print(len(augmented['data']), ' examples processed.', end='\r')
    with open('./data/augmented_train.json', 'w') as f:
        dump(augmented, f)
        
        
    

def swap(sentence, answers, n):
    sentence = sentence.split(' ')
    if len(sentence) < 2:
        return ' '.join(word for word in sentence)
    new = sentence.copy()
    for _ in range(n):
        idx1 = random.randint(0, len(sentence)-2)
        idx2 = random.randint(0, len(sentence)-2)
        count = 0
        while (idx1 == idx2) or new[idx1] in answers or new[idx2] in answers or (new[idx1].strip(string.punctuation) in answers) or (new[idx2].strip(string.punctuation) in answers):
            if count > 3: break
            idx1 = random.randint(0, len(sentence)-1)
            idx2 = random.randint(0, len(sentence)-1)
            count += 1
        if count <= 3: new[idx1], new[idx2] = new[idx2], new[idx1]
    new_str = ' '.join(word for word in new)
    return new_str

def delete(sentence, answers, p):
    sentence = sentence.split(' ')
    if len(sentence) == 1: return sentence[0]
    new = []
    for word in sentence:
        if random.random() > p or (word.strip(string.punctuation) in answers) or word in answers or word == sentence[-1]:
            new.append(word)
    if len(new) == 0: return sentence[0]
    new_str = ' '.join(word for word in new)
    return new_str

if __name__ == '__main__':
    main(get_train_args())
