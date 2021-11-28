'''
This script is to convert the NTY dataset from the version preprocessed by Jiaqi to TACRED-like json version.
As the version preprocessed by Jiaqi used str rather than json for each instance.
'''

import json

import json
import spacy
import re
from spacy.tokenizer import Tokenizer

STOP = '.'
# nlp = spacy.load("en_core_web_sm")
# nlp.tokenizer = Tokenizer(nlp.vocab) # add this one for no split for dog's, you're

import spacy_stanza
# Call this if stanza has missing resources
import stanza
# stanza.download('en')
nlp = spacy_stanza.load_pipeline("en", tokenize_pretokenized=True)
# nlp = stanza.Pipeline('en')
# nlp.tokenizer = Tokenizer(nlp.vocab) # add this one for no split for dog's, you're

PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1
NER_OTHER = 'O'
#NER_OTHER = 'NER_OTHER'

NYT = 'nyt'
NYT24 = 'nyt24'
NYT29 = 'nyt29'
WEBNLG = 'webnlg'
ONTONOTES = 'NOTONOTES'
ATIS = 'atis'
FewRel = 'fewrel'
WIKI80 = 'wiki80'
MAVEN = 'maven_causal'


DATASET = ATIS

def get_list(input_list):
    new_list = []
    for i, t in enumerate(input_list):
        if i == 0:
            new_list.append(t[2:-2])
        elif i == len(input_list) - 1:
            new_list.append(t[1:-3])
        else:
            new_list.append(t[1:-2])
    return new_list

def get_list_lower(input_list):
    new_list = []
    for i, t in enumerate(input_list):
        if i == 0:
            new_list.append((t[2:-2]).lower())
        elif i == len(input_list) - 1:
            new_list.append((t[1:-3]).lower())
        else:
            new_list.append((t[1:-2]).lower())
    return new_list

def get_list_digits(input_list):
    new_list = []
    for i, t in enumerate(input_list):
        if i == 0:
            new_list.append(int(t[1:-1]))
        elif i == len(input_list) - 1:
            new_list.append(int(t[:-2]))
        else:
            new_list.append(int(t[:-1]))
    return new_list

def convert_json(input_file, output_file):

    print("converting the file {}".format(input_file))
    obj_type_set = set()
    subj_type_set = set()
    rel_type_set = set()

    new_json = []

    ignored = 0

    tmp_rel_list = []

    with open(input_file, 'rb') as f:
        data = json.load(f)
        print("len of the {} is {}".format(input_file, len(data)))
        for no, l in enumerate(data):
            inst = json.loads(l)
            # if no > 2000:
            #     break

            # tmp_rel_list.append(inst['relation'])
            # continue
            if no % 1000 == 0:
                print("processing {}th instance".format(no))
                #print("ignored {} instances".format(ignored))

            l_json = {}
            l_json["id"] = inst['id']

            l_json["relation"] = inst['relation']

            if type(inst['relation']) is list:
                for r in l_json["relation"]:
                    rel_type_set.add(r)
            else:
                rel_type_set.add(l_json["relation"])

            l_json["token"] = inst['token']

            pos_tags = []
            dep_heads = []
            for word in nlp(' '.join(l_json["token"])):
                pos_tags.append(word.tag_)

            root_index = 0
            for word in nlp(' '.join(l_json["token"])):
                if word.i == word.head.i:
                    dep_heads.append(0)
                    root_index = word.i
                else:
                    if word.head.i + 1 == 0:
                        print("debug")
                    dep_heads.append(word.head.i + 1)

            # to avoid too many zero for head in the tree
            for head_no, head in enumerate(dep_heads):
                if head == 0:
                    if head_no != root_index:
                        dep_heads[head_no] = root_index + 1

            flag = len(l_json["token"]) == len(pos_tags) == len(dep_heads)

            if not flag:
                print("len error")
                continue

            l_json["subj_start"] = inst['subj_start']
            l_json["subj_end"] = inst['subj_end']
            l_json["obj_start"] = inst['obj_start']
            l_json["obj_start"] = inst['obj_start']
            l_json["obj_end"] = inst['obj_end']
            l_json["obj_end"] = inst['obj_end']

            if l_json["subj_start"] > l_json["subj_end"] or l_json["obj_start"] > l_json["obj_end"]:
                print("subj and obj position error")

            l_json["stanford_pos"] = pos_tags #get_list(pos_tags)

            ner_tags = inst['stanford_ner']
            for tag_no, tag in enumerate(ner_tags):
                if tag == '':
                   ner_tags[tag_no] = NER_OTHER

            l_json["stanford_ner"] = ner_tags

            subj_type = inst['subj_type']
            obj_type = inst['obj_type']

            if type (obj_type) is list or type(subj_type) is list:
                for t_no, (ot, st) in enumerate(zip(obj_type, subj_type)):
                    if ot == '' or ot == 'O':
                        obj_type[t_no] = NER_OTHER
                    if st == '' or st == 'O':
                        subj_type[t_no] = NER_OTHER
            else:
                if obj_type == '' or obj_type == 'O':
                    obj_type = NER_OTHER
                if subj_type == '' or subj_type == 'O':
                    subj_type = NER_OTHER

            l_json["obj_type"] = subj_type
            l_json["subj_type"] = obj_type

            if type(subj_type) is list or type(obj_type) is list:
                for ot, st in zip(l_json["subj_type"], l_json["obj_type"]):
                    obj_type_set.add(ot)
                    subj_type_set.add(st)
            else:
                obj_type_set.add(l_json["subj_type"])
                subj_type_set.add(l_json["obj_type"])

            l_json["stanford_deprel"] = [rel.lower() for rel in inst['stanford_deprel']]

            l_json["stanford_head"] = dep_heads#get_list_digits(head)

            if 'trigger' in inst:
                if 'trigger_id' in inst:
                    l_json["trigger"] = inst['trigger']
                    l_json["trigger_id"] = inst['trigger_id']
                else:
                    exit(1)
                assert len(l_json["stanford_pos"]) == len(l_json["trigger"]) == len(l_json["trigger_id"])

            assert  len(l_json["stanford_pos"]) == len(l_json["stanford_ner"]) == len(l_json["stanford_deprel"]) == len(l_json["stanford_head"])

            new_json.append(l_json)

    from collections import Counter
    rel_list = Counter(tmp_rel_list)

    print(rel_list)

    with open(output_file, 'w') as f:
        json.dump(new_json, f)

    print("inst num for {} is {}".format(output_file, len(new_json)))

    subj_dic = {}
    obj_dic = {}
    rel_dic = {}
    ner_dic = {}

    subj_dic[PAD_TOKEN] = 0
    subj_dic[UNK_TOKEN] = 1
    obj_dic[PAD_TOKEN] = 0
    obj_dic[UNK_TOKEN] = 1
    ner_dic[PAD_TOKEN] = 0
    ner_dic[UNK_TOKEN] = 1

    for set_no, set_value in enumerate(subj_type_set):
        subj_dic[set_value] = set_no + 2

    for set_no, set_value in enumerate(obj_type_set):
        obj_dic[set_value] = set_no + 2

    for set_no, set_value in enumerate(rel_type_set):
        rel_dic[set_value] = set_no

    ner_set = subj_type_set.union(obj_type_set)

    for set_no, set_value in enumerate(ner_set):
        ner_dic[set_value] = set_no + 2

    print("SUBJ_NER_TO_ID = {}".format(subj_dic))
    print("OBJ_NER_TO_ID = {}".format(obj_dic))
    print('NER_TO_ID = {}'.format(ner_dic))
    #print(rel_dic)

    return rel_type_set

if DATASET == NYT:
    train_original = 'dataset/nyt/train_jiaqi.json'
    train_file = 'dataset/nyt/train.json'

    dev_original = 'dataset/nyt/dev_jiaqi.json'
    dev_file = 'dataset/nyt/dev.json'

    test_original = 'dataset/nyt/test_jiaqi.json'
    test_file = 'dataset/nyt/test.json'

elif DATASET == WEBNLG:
    train_original = 'dataset/webnlg/train_jiaqi.json'
    train_file = 'dataset/webnlg/train.json'

    dev_original = 'dataset/webnlg/dev_jiaqi.json'
    dev_file = 'dataset/webnlg/dev.json'

    test_original = 'dataset/webnlg/test_jiaqi.json'
    test_file = 'dataset/webnlg/test.json'

elif DATASET == NYT24:
    train_original = 'dataset/nyt24/train_jiaqi.json'
    train_file = 'dataset/nyt24/train.json'

    dev_original = 'dataset/nyt24/dev_jiaqi.json'
    dev_file = 'dataset/nyt24/dev.json'

    test_original = 'dataset/nyt24/test_jiaqi.json'
    test_file = 'dataset/nyt24/test.json'

elif DATASET == NYT29:
    train_original = 'dataset/nyt29/train_jiaqi.json'
    train_file = 'dataset/nyt29/train.json'

    dev_original = 'dataset/nyt29/dev_jiaqi.json'
    dev_file = 'dataset/nyt29/dev.json'

    test_original = 'dataset/nyt29/test_jiaqi.json'
    test_file = 'dataset/nyt29/test.json'

elif DATASET == ONTONOTES:
    train_original = 'dataset/ontonotes/train_jiaqi.json'
    train_file = 'dataset/ontonotes/train.json'

    dev_original = 'dataset/ontonotes/dev_jiaqi.json'
    dev_file = 'dataset/ontonotes/dev.json'

    test_original = 'dataset/ontonotes/test_jiaqi.json'
    test_file = 'dataset/ontonotes/test.json'

elif DATASET == ATIS:
    train_original = 'dataset/atis/train_jiaqi.json'
    train_file = 'dataset/atis/train.json'

    dev_original = 'dataset/atis/dev_jiaqi.json'
    dev_file = 'dataset/atis/dev.json'

    test_original = 'dataset/atis/test_jiaqi.json'
    test_file = 'dataset/atis/test.json'

elif DATASET == FewRel:
    train_original = 'dataset/fewrel/train_jiaqi.json'
    train_file = 'dataset/fewrel/train.json'

    dev_original = 'dataset/fewrel/dev_jiaqi.json'
    dev_file = 'dataset/fewrel/dev.json'

    test_original = 'dataset/fewrel/test_jiaqi.json'
    test_file = 'dataset/fewrel/test.json'

elif DATASET == WIKI80:
    train_original = 'dataset/wiki80/train_jiaqi.json'
    train_file = 'dataset/wiki80/train.json'

    dev_original = 'dataset/wiki80/dev_jiaqi.json'
    dev_file = 'dataset/wiki80/dev.json'

    test_original = 'dataset/wiki80/test_jiaqi.json'
    test_file = 'dataset/wiki80/test.json'

elif DATASET == MAVEN:
    train_original = 'dataset/maven_causal/train_jiaqi.json'
    train_file = 'dataset/maven_causal/train.json'

    dev_original = 'dataset/maven_causal/dev_jiaqi.json'
    dev_file = 'dataset/maven_causal/dev.json'

    test_original = 'dataset/maven_causal/test_jiaqi.json'
    test_file = 'dataset/maven_causal/test.json'

train_rel_set = convert_json(train_original, train_file)
dev_rel_set = convert_json(dev_original, dev_file)
test_rel_set = convert_json(test_original, test_file)

u_set = train_rel_set.union(dev_rel_set)
u_set = u_set.union(test_rel_set)

all_rel_dic = {}

for set_no, set_value in enumerate(u_set):
    all_rel_dic[set_value] = set_no

print("LABEL_TO_ID = {} ".format(all_rel_dic))