

import json
import random
import torch
import numpy as np

from utils import constant, helper, vocab

class NERDataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, charvocab, evaluation=False):
        

        self.max_char_len = 20
        
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.charvocab = charvocab
        self.eval = evaluation
        self.label2id = constant.LABEL_TO_ID

        with open(filename) as infile:
            data = json.load(infile)
        self.raw_data = data
        data = self.preprocess(data, vocab, opt)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.num_examples = len(data)
        self.goldner = []
        for d in data:
            self.goldner += d[2]
        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        i = 0
        for d in data:
            i += 1
            tokens = list(d['token'])
            chars_seq = []
            for token in tokens:
                if len(token) <= self.max_char_len:
                    chars = map_to_ids(token, self.charvocab.word2id) + [0 for i in range(self.max_char_len-len(token))]
                else:
                    chars = map_to_ids(token[0:self.max_char_len],self.charvocab.word2id)
                chars_seq.append(chars)

            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            tokens = map_to_ids(tokens, vocab.word2id)

            l = len(tokens)
            ner = map_to_ids(d['stanford_ner'], constant.LABEL_TO_ID)
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            head = [int(x) for x in d['stanford_head']]
            subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
            obj_positions = get_positions(d['obj_start'], d['obj_end'], l)

            processed += [(tokens,chars_seq, ner, pos, deprel, head, subj_positions, obj_positions)]
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.goldner

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))

        lens = [len(x) for x in batch[0]]

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size)
        masks = torch.gt(words, 0)
        chars_seq = get_long_tensor_char(batch[1], batch_size)

        labels = get_long_tensor_label(batch[2],batch_size)
        pos = get_long_tensor(batch[3], batch_size)
        deprel = get_long_tensor(batch[4], batch_size)
        head = get_long_tensor_head(batch[5], batch_size)
        subj_positions = get_long_tensor(batch[6], batch_size)
        obj_positions = get_long_tensor(batch[7], batch_size)

        bert_input=None
        return (words, masks, chars_seq, pos, deprel, head, subj_positions, obj_positions, bert_input,labels)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)




def map_to_ids(tokens, vocab):
    ids = [vocab[t] if t in vocab else constant.UNK_ID for t in tokens]
    return ids

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens
def get_long_tensor_head(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        s1=s
        for h in range(len(s)-1):
            if s[h]==0 and s[h+1]!=0:
                s[h]=h
        for h in range(len(s)):
            if s[h]!=0 and s1[h]==s[h]:
                s[h]-=1
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens

def get_long_tensor_label(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(-100)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
    return tokens
def get_long_tensor_char(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len, 20).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        for j, x in enumerate(s):
            tokens[i,j,:] = torch.LongTensor(x)
    return tokens

def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def word_dropout(tokens, dropout):
    """ Randomly dropout tokens (IDs) and replace them with <UNK> tokens. """
    return [constant.UNK_ID if x != constant.UNK_ID and np.random.random() < dropout \
            else x for x in tokens]

