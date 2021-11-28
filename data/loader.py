"""
Data loader for TACRED json files.
"""

import json
import random
import torch
import numpy as np
import math

from utils import constant, helper, vocab
from utils.ClassAwareSampler import ClassAwareSampler

# for bert
import transformers
from transformers import BertForTokenClassification, BertConfig, BertTokenizer
additional_special_tokens = ["[E11]", "[E12]", "[E21]", "[E22]"]

class DataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False, mr = False):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.label2id = constant.LABEL_TO_ID
        self.mr = mr  # multiple relations
        self.filename = filename

        with open(filename) as infile:
            data = json.load(infile)
        self.raw_data = data
        data = self.preprocess(data, vocab, opt)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])

        if self.mr:
            labels = [[self.id2label[r] for r in d[-1]]for d in data]
            self.labels = []
            for l_vector in labels:
                self.labels += l_vector
        else:

            self.labels = [self.id2label[d[-1]] for d in data]

        self.num_examples = len(data)

        ### baseline
        if self.opt.get('cbs', False) and not evaluation:
            sampler = ClassAwareSampler(data)
            new_data = [data[next(iter(sampler))] for i in range(self.num_examples)]
            print("Use class balanced sampling")
            data = new_data

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, vocab, opt):
        """ Preprocess the data and convert to ids. """
        processed = []
        max_len = 0
        max_index = 0
        ignore_num = 0
        subj_obj_eq = 0

        PAD_TOKEN = '<PAD>'
        PAD_ID = 0
        UNK_TOKEN = '<UNK>'
        UNK_ID = 1
        NER_OTHER = 'O'

        type_dic = {}
        count = 0
        type_set = set()
        for no, (key, value) in enumerate(constant.LABEL_TO_ID.items()):
             type_set.add(key)

        for no, value in enumerate(type_set):
            type_dic[value] = no

        # print(type_dic)
        print(self.label2id)

        # for no, (key, value) in enumerate(constant.LABEL_TO_ID.items()):
        #      type_set.add(value)
        #      ner_type_dic[key] = no
            # if key == PAD_TOKEN or key == UNK_TOKEN:
            #     continue
            # elif key == NER_OTHER:
            #     event_type_dic[key] = count
            #     count += 1
            # else:
            #     b_key = 'B-'+key
            #     i_key = 'I-'+key
            #     event_type_dic[b_key] = count
            #     count += 1
            #     event_type_dic[i_key] = count
            #     count += 1



        # print(ner_type_dic)

        for d_no, d in enumerate(data):
            #
            # if d_no > 1000:
            #     break

            tokens = list(d['token'])
            tmp_len = len(tokens)
            if d['subj_start'] == d['obj_start'] and d['subj_end'] == d['obj_end']:
                # print("subj and obj are the same")
                subj_obj_eq += 1
                continue
            if tmp_len > opt['max_len']:
                print("len is {} and index is {}".format(tmp_len, d_no))
                ignore_num += 1
                continue
            # if tmp_len > max_len:
            #     max_len = tmp_len
            #     max_index = d_no
                 
            # instance selection
            # if self.opt['instance_class'] is not None:
                # d['subj_start'] = max(0, d['subj_start']-1)
                # d['obj_start'] = max(0, d['obj_start']-1)
            if opt['lower']:
                tokens = [t.lower() for t in tokens]
            # anonymize tokens
            if self.mr:
                m_ss, m_se = d['subj_start'], d['subj_end']
                m_os, m_oe = d['obj_start'], d['obj_end']
                for ss, se, st in zip(m_ss, m_se, d['subj_type']):
                    tokens[int(ss):int(se) + 1] = ['SUBJ-' + st] * (int(se) - int(ss) + 1)  # very critical
                for os, oe, ot in zip(m_os, m_oe, d['obj_type']):
                    tokens[int(os):int(oe) + 1] = ['OBJ-' + ot] * (int(oe) - int(os) + 1)  # very ctritical
            else:
                ss, se = d['subj_start'], d['subj_end']
                os, oe = d['obj_start'], d['obj_end']
                tokens[ss:se+1] = ['SUBJ-'+d['subj_type']] * (se-ss+1) # very critical
                tokens[os:oe+1] = ['OBJ-'+d['obj_type']] * (oe-os+1) # very ctritical

            tokens = map_to_ids(tokens, vocab.word2id)
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            head = [int(x) for x in d['stanford_head']]
            assert any([x == 0 for x in head])
            assert 0 in head
            # if  any([x == 0 for x in head]):
            #     print("ignored due to 0 for head")
            #     continue
            l = len(tokens)
            if self.mr:
                assert len(d['subj_start']) == len(d['subj_end']) == len(d['obj_start']) == len(d['obj_end'])
                num_rel = len(d['subj_start'])
                subj_positions = [[1] * l for r in range(num_rel)]
                obj_positions = [[1] * l for r in range(num_rel)]
                for no, (starts, ends) in enumerate(zip(d['subj_start'], d['subj_end'])):
                    # for s, e in zip(str(starts), str(ends)):
                    #     s = int(s)
                    #     e = int(e)
                    #     if e >= l:
                    #         s = s - 1
                    #         e = l - 1
                    #         print("{}th data, remedy the index to avoid out of boundary".format(d_no))
                    #     subj_positions[no][s] = 0
                    #     subj_positions[no][e] = 0
                    # NOTE: changed:
                    s = int(starts)
                    e = int(ends)
                    subj_positions[no][s:e+1] = [0]*(e-s+1)
                for no, (starts, ends) in enumerate(zip(d['obj_start'], d['obj_end'])):
                    # for s, e in zip(str(starts), str(ends)):
                    #     s = int(s)
                    #     e = int(e)
                    #     if e >= l:
                    #         s = s - 1
                    #         e = l - 1
                    #         print("{}th data, remedy the index to avoid out of boundary".format(d_no))
                    #     obj_positions[no][s] = 0
                    #     obj_positions[no][e] = 0
                    # NOTE: changed:
                    s = int(starts)
                    e = int(ends)
                    obj_positions[no][s:e+1] = [0]*(e-s+1)
                relation = [self.label2id[r] for r in d['relation']]  # For calculate the score
                re_size = len(self.label2id)
                bce_relation = []
                for r in relation:
                    tag = [0.] * re_size
                    tag[r] = 1.
                    bce_relation.append(tag)
                rel_num = len(relation)
                subj_type = [constant.SUBJ_NER_TO_ID[t] for t in d['subj_type']]
                obj_type = [constant.OBJ_NER_TO_ID[t] for t in d['obj_type']]
            else:
                subj_positions = get_positions(d['subj_start'], d['subj_end'], l)
                obj_positions = get_positions(d['obj_start'], d['obj_end'], l)
                if len(subj_positions) != len(obj_positions):
                    print("debug")
                relation = self.label2id[d['relation']]
                subj_type = [constant.SUBJ_NER_TO_ID[d['subj_type']]]
                obj_type = [constant.OBJ_NER_TO_ID[d['obj_type']]]

            # if len(tokens) < 15 and self.label2id[d['relation']] in [4, 21, 19, 2, 9]:
            if self.opt['instance_class'] is not None:
                if 'test' in self.filename and len(tokens) <= 10 and self.label2id[d['relation']] in [self.opt['instance_class']]:
                # if 'test' in self.filename and len(tokens) <= 10 and self.label2id[d['relation']] in [self.opt['instance_class']] and 'Modena' in d['token']:
                # if len(tokens) <= 20 and self.label2id[d['relation']] in [self.opt['instance_class']]:
                    print(len(processed), d['token'], d['relation'], '_'.join(d['token'][d['subj_start']:d['subj_end']+1]), \
                        '_'.join(d['token'][d['obj_start']:d['obj_end']+1]))
                    processed += [(tokens, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, relation)]
                    continue
                else: 
                    continue

            if self.mr:
                processed += [(tokens, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, rel_num, bce_relation, relation)]
            else:
                processed += [(tokens, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, relation)]
            

        #print("max len is {} and index is {}".format(max_len, max_index))
        print("ignored num for instances longer than max len {}".format(ignore_num))
        print("the num of sub and obj equal and ignored {}".format(subj_obj_eq))
        print("num of total instances:", len(processed))
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

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
        if self.mr:
            assert len(batch) == 12
        else:
            assert len(batch) == 10

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size)
        masks = torch.eq(words, 0)
        pos = get_long_tensor(batch[1], batch_size)
        ner = get_long_tensor(batch[2], batch_size)
        deprel = get_long_tensor(batch[3], batch_size)
        head = get_long_tensor(batch[4], batch_size)

        if self.mr:
            subj_positions = get_tensor(batch[5], batch_size)
            obj_positions = get_tensor(batch[6], batch_size)
            subj_type = get_long_tensor(batch[7], batch_size)
            obj_type = get_long_tensor(batch[8], batch_size)
            rels = get_tensor(batch[10], batch_size)#relation labels  (BCE) for training

            max_rel_num = max(batch[9])
            rel_masks = torch.tensor([[1] * l + [0] * (max_rel_num - l) for l in batch[9]]).float()
            rel_nums = batch[9]  # the num of relation for each instance in a batch

            return (words, masks, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, rels, rel_masks, rel_nums, orig_idx)
        else:
            subj_positions = get_long_tensor(batch[5], batch_size)
            obj_positions = get_long_tensor(batch[6], batch_size)
            subj_type = get_long_tensor(batch[7], batch_size)
            obj_type = get_long_tensor(batch[8], batch_size)
            rels = torch.LongTensor(batch[9])

            return (words, masks, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, rels, orig_idx)

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

def get_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    max_token_len = max(len(x) for x in tokens_list)
    if max_token_len == 0:
        tokens = torch.LongTensor(batch_size, max_token_len).fill_(constant.PAD_ID)
        for i, s in enumerate(tokens_list):
            tokens[i, :len(s)] = torch.LongTensor(s)
        return tokens
    max_rel_num = max(len(rels) for inst in tokens_list for rels in inst)
    tokens = torch.LongTensor(batch_size, max_token_len, max_rel_num).fill_(constant.PAD_ID)
    for i, x in enumerate(tokens_list): # for the ith relations in the batch
        for j, s in enumerate(x):# for the jth sentneces in the ith instance
            tokens[i, j, :len(s)] = torch.LongTensor(s)
    return tokens


def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list):
        tokens[i, :len(s)] = torch.LongTensor(s)
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

class BertDataLoader(object):
    """
    Load data from json files, preprocess and prepare batches.
    """
    def __init__(self, filename, batch_size, opt, vocab, evaluation=False, mr = False):
        self.batch_size = batch_size
        self.opt = opt
        self.vocab = vocab
        self.eval = evaluation
        self.label2id = constant.LABEL_TO_ID
        self.mr = mr  # multiple relations

        # for bert
        self.bert_model = opt['bert_model']
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=self.opt['lower'], 
                additional_special_tokens=additional_special_tokens)

        with open(filename) as infile:
            data = json.load(infile)
        self.raw_data = data
        data = self.preprocess(data, opt, vocab)

        # shuffle for training
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.id2label = dict([(v,k) for k,v in self.label2id.items()])

        self.labels = [self.id2label[d[9]] for d in data]
            

        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data
        print("{} batches created for {}".format(len(data), filename))

    def preprocess(self, data, opt, vocab):
        """ Preprocess the data and convert to ids. """
        processed = []
        max_len = opt['max_len']
        max_index = 0
        ignore_num = 0
        subj_obj_eq = 0

        # NOTE: limit max data point
        num_22 = 0

        for d_no, d in enumerate(data):
            
            relation = self.label2id[d['relation']]
            if relation == 22 or relation == '22':
                if num_22 >= 10000:
                    continue
                num_22 += 1

            tokens = list(d['token'])
            tmp_len = len(tokens)
            if d['subj_start'] == d['obj_start'] and d['subj_end'] == d['obj_end']:
                # print("subj and obj are the same")
                subj_obj_eq += 1
                continue
            # NOTE: removed
            # if tmp_len > opt['max_len']:
            #     print("len is {} and index is {}".format(tmp_len, d_no))
            #     ignore_num += 1
            #     continue
            # if tmp_len > max_len:
            #     max_len = tmp_len
            #     max_index = d_no
            #     continue
            if opt['lower'] or 'uncased' in self.bert_model:
                tokens = [t.lower() for t in tokens]
            original_tokens = [i for i in tokens]

            # anonymize tokens
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']
            tokens[ss:se+1] = ['SUBJ-'+d['subj_type']] * (se-ss+1) # very critical
            tokens[os:oe+1] = ['OBJ-'+d['obj_type']] * (oe-os+1) # very ctritical
            tokens = map_to_ids(tokens, vocab.word2id)
            pos = map_to_ids(d['stanford_pos'], constant.POS_TO_ID)
            ner = map_to_ids(d['stanford_ner'], constant.NER_TO_ID)
            deprel = map_to_ids(d['stanford_deprel'], constant.DEPREL_TO_ID)
            head = [int(x) for x in d['stanford_head']]
            assert any([x == 0 for x in head])
            assert 0 in head
            # if  any([x == 0 for x in head]):
            #     print("ignored due to 0 for head")
            #     continue
            l = len(tokens)
            if self.mr:
                assert len(d['subj_start']) == len(d['subj_end']) == len(d['obj_start']) == len(d['obj_end'])
                num_rel = len(d['subj_start'])
                subj_positions = [[1] * l for r in range(num_rel)]
                obj_positions = [[1] * l for r in range(num_rel)]
                for no, (starts, ends) in enumerate(zip(d['subj_start'], d['subj_end'])):
                    for s, e in zip(starts, ends):
                        if e >= l:
                            s = s - 1
                            e = l - 1
                            print("{}th data, remedy the index to avoid out of boundary".format(d_no))
                        subj_positions[no][s] = 0
                        subj_positions[no][e] = 0
                for no, (starts, ends) in enumerate(zip(d['obj_start'], d['obj_end'])):
                    for s, e in zip(starts, ends):
                        if e >= l:
                            s = s - 1
                            e = l - 1
                            print("{}th data, remedy the index to avoid out of boundary".format(d_no))
                        obj_positions[no][s] = 0
                        obj_positions[no][e] = 0
                flat_relation = [self.label2id[r] for r in d['relation']]  # For calculate the score
                re_size = len(self.label2id)
                bce_relation = []
                for r in flat_relation:
                    tag = [0.] * re_size
                    tag[r] = 1.
                    bce_relation.append(tag)

            else:
                subj_start, subj_end = d['subj_start'], d['subj_end']
                obj_start, obj_end = d['obj_start'], d['obj_end']
                l = len(tokens)

                # This is should not be used in bert model
                subj_positions = get_positions(subj_start, subj_end, l)
                obj_positions = get_positions(obj_start, obj_end, l)
                if len(subj_positions) != len(obj_positions):
                    print("debug")

                ##### NOTE: context cf added ####
                # from model.counterfactual import get_shortest_path_tmp, get_neighbor
                # neighbors = get_neighbor(head, l, subj_positions, obj_positions)
                # subj_indices = list(range(subj_start, subj_end+1))
                # obj_indices = list(range(obj_start, obj_end))
                # context = list(set(neighbors+subj_indices+obj_indices))
                # context_cf_tokens = [original_tokens[i] for i in range(len(original_tokens)) if i not in context]
                # adj = get_adj(head)
                # path = get_shortest_path_tmp(adj, subj_start, subj_end, obj_start, obj_end)
                
                ##### NOTE: BERT Added ####
                subject_first = (subj_start < obj_start)
                if subject_first:  # first encountered entity is subject
                    new_tokens = original_tokens[:subj_start] + ["[E11]"] + original_tokens[subj_start:subj_end+1] + ["[E12]"] + \
                        original_tokens[subj_end+1:obj_start] + ["[E21]"] + original_tokens[obj_start:obj_end+1] + ["[E22]"] + \
                        original_tokens[obj_end+1:]
                    subj_start += 1
                    subj_end += 1
                    obj_start += 3
                    obj_end += 3
                else:
                    new_tokens = original_tokens[:obj_start] + ["[E21]"] + original_tokens[obj_start:obj_end+1] + ["[E22]"] + \
                        original_tokens[obj_end+1:subj_start] + ["[E11]"] + original_tokens[subj_start:subj_end+1] + ["[E12]"] + \
                        original_tokens[subj_end+1:]
                    subj_start += 3
                    subj_end += 3
                    obj_start += 1
                    obj_end += 1
                original_tokens = new_tokens

                bert_tokens = self.tokenizer.tokenize(' '.join(new_tokens), is_pretokenized=True)
                bert_tokens = _truncate_seq(bert_tokens, max_len-2)
                # bert_tokens = ['[CLS]'] + bert_tokens + ['[SEP]']
                if '[E22]' not in bert_tokens or '[E12]' not in bert_tokens:
                    print("Index {} has entity removed from truncation, skipped".format(d_no))
                    continue
                else:
                    e11_p = bert_tokens.index("[E11]")+1
                    e12_p = bert_tokens.index("[E12]")
                    e21_p = bert_tokens.index("[E21]")+1
                    e22_p = bert_tokens.index("[E22]")

                    # CLS padding
                    e11_p += 1
                    e12_p += 1
                    e21_p += 1
                    e22_p += 1

                    e1_mask = [0 for i in range(max_len)]
                    e2_mask = [0 for i in range(max_len)]
                    
                    for i in range(e11_p, e12_p):
                        # e1_mask[i] = 1/(e12_p-e11_p)
                        e1_mask[i] = 1
                    for i in range(e21_p, e22_p):
                        # e2_mask[i] = 1/(e22_p-e21_p)
                        e2_mask[i] = 1
                bert_token_ids = self.tokenizer.convert_tokens_to_ids(bert_tokens)
                if not (len(bert_token_ids) > 0 and len(bert_token_ids) <= max_len -2):
                    # print("Ill processed tokens:", original_tokens)
                    print("Ill processed bert tokens:", bert_tokens)
                    continue
                ##### END

            subj_type = [constant.SUBJ_NER_TO_ID[d['subj_type']]]
            obj_type = [constant.OBJ_NER_TO_ID[d['obj_type']]]
            relation = self.label2id[d['relation']]
            # for bert: added original tokens
            # processed += [(tokens, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, relation)]
            processed += [(tokens, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, relation, original_tokens, 
                        e1_mask, e2_mask, bert_token_ids)]
        #print("max len is {} and index is {}".format(max_len, max_index))
        print("ignored num for instances longer than max len {}".format(ignore_num))
        print("the num of sub and obj equal and ignored {}".format(subj_obj_eq))
        print("num 22", num_22)
        return processed

    def gold(self):
        """ Return gold labels as a list. """
        return self.labels

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
        # assert len(batch) == 11

        # sort all fields by lens for easy RNN operations
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = sort_all(batch, lens)

        # word dropout
        if not self.eval:
            words = [word_dropout(sent, self.opt['word_dropout']) for sent in batch[0]]
        else:
            words = batch[0]

        # convert to tensors
        words = get_long_tensor(words, batch_size)
        masks = torch.eq(words, 0)
        pos = get_long_tensor(batch[1], batch_size)
        ner = get_long_tensor(batch[2], batch_size)
        deprel = get_long_tensor(batch[3], batch_size)
        head = get_long_tensor(batch[4], batch_size)
        subj_positions = get_long_tensor(batch[5], batch_size)
        obj_positions = get_long_tensor(batch[6], batch_size)
        subj_type = get_long_tensor(batch[7], batch_size)
        obj_type = get_long_tensor(batch[8], batch_size)

        rels = torch.LongTensor(batch[9])
        
        # for bert
        original_tokens = batch[10]
        e1_mask = batch[11]
        e2_mask = batch[12]
        encoded_inputs = batch[13]
        lens = [len(i) for i in original_tokens]
        
        # encoded_inputs = []
        # batch_token_lens = []
        # for sent_tokens in original_tokens:
        #     tokens = []
        #     pieces = []
        #     token_lens = []
        #     for token_text in sent_tokens:
        #         token_pieces = [p for p in self.tokenizer.tokenize(token_text) if p]
        #         if len(token_pieces) == 0:
        #             continue
        #         tokens.append(token_text)
        #         pieces.extend(token_pieces)
        #         token_lens.append(len(token_pieces))
        #     # truncate to maxlen if necessary

        #     # pad word pieces with special tokens
        #     piece_idxs = self.tokenizer.encode(pieces,
        #                                   add_special_tokens=True)
        #     encoded_inputs.append(piece_idxs)
        #     batch_token_lens.append(token_lens)
        bert_inputs = self.tokenizer.batch_encode_plus(encoded_inputs, 
                        return_tensors='pt', 
                        is_pretokenized=True,
                        max_length=self.opt['max_len'],
                        padding='max_length')
                        # padding=True)
        # bert_inputs['token_lens'] = batch_token_lens
        bert_inputs['e1_mask'] = get_long_tensor(e1_mask, batch_size)
        bert_inputs['e2_mask'] = get_long_tensor(e2_mask, batch_size)

        return (words, masks, pos, ner, deprel, head, subj_positions, obj_positions, subj_type, obj_type, rels, orig_idx, bert_inputs)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)


def _truncate_seq(tokens_a, max_length):
    """Truncates a sequence """
    tmp = tokens_a[:max_length]
    if ("[E12]" in tmp) and ("[E22]" in tmp):
        return tmp
    else:
        e11_p = tokens_a.index("[E11]")
        e12_p = tokens_a.index("[E12]")
        e21_p = tokens_a.index("[E21]")
        e22_p = tokens_a.index("[E22]")
        start = min(e11_p, e12_p, e21_p, e22_p)
        end = max(e11_p, e12_p, e21_p, e22_p)
        if end-start+1 > max_length:
            remaining_length = max_length - (e12_p-e11_p+1) - (e22_p-e21_p+1)  
            first_addback = math.floor(remaining_length/2)
            second_addback = remaining_length - first_addback
            if start == e11_p:
                new_tokens = tokens_a[e11_p: e12_p+1+first_addback] + tokens_a[e21_p-second_addback:e22_p+1]
            else:
                new_tokens = tokens_a[e21_p: e22_p+1+first_addback] + tokens_a[e11_p-second_addback:e12_p+1]
            return new_tokens
        else:
            new_tokens = tokens_a[start:end+1]
            remaining_length = max_length - len(new_tokens)
            if start < remaining_length:  # add sentence beginning back
                new_tokens = tokens_a[:start] + new_tokens 
                remaining_length -= start
            else:
                new_tokens = tokens_a[start-remaining_length:start] + new_tokens
                return new_tokens

            # still some room left, add sentence end back
            new_tokens = new_tokens + tokens_a[end+1:end+1+remaining_length]
            return new_tokens
            
def get_event_long_tensor(tokens_list, batch_size, ):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    t = torch.LongTensor(batch_size, token_len).fill_(constant.PAD_ID)
    for i, s in enumerate(tokens_list): # for the ith instance in the batch
        t[i, :len(s), :] = torch.LongTensor(s)
    return t
