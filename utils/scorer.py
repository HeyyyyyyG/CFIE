#!/usr/bin/env python

"""
Score the predictions with gold labels, using precision, recall and F1 metrics.
"""

import argparse
import sys
from collections import Counter
from utils import constant
import numpy as np
#NO_RELATION = "no_relation"

TACRED = 'tacred'
NYT = 'nyt'
NYT24 = 'nyt24'
NYT29 = 'nyt29'
ACE = 'ace2005'
WEBNLG = 'webnlg'
FewRel = 'fewrel'
Wiki80 = 'wiki80'

def parse_arguments():
    parser = argparse.ArgumentParser(description='Score a prediction file using the gold labels.')
    parser.add_argument('gold_file', help='The gold relation file; one relation per line')
    parser.add_argument('pred_file', help='A prediction file; one relation per line, in the same order as the gold file.')
    args = parser.parse_args()
    return args

def calc_mean(array, num_inst, last = False):
    recall = 0
    pre = 0
    f1 = 0
    for result in array:
        recall += result[0]
        pre += result[1]
        f1 += result[2]

    if len(array) > 0:
        meanrecall = recall / len(array)
        meanpre = pre / len(array)
        meanf1 = f1 / len(array)
    else:
        meanpre = 0
        meanrecall = 0
        meanf1 = 0

    if last:
        print('Mean Performance for Instances largers than {} : P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(num_inst,
            meanpre * 100.0, meanrecall * 100.0, meanf1 * 100.0))
    else:
        print('Mean Performance for Instances less and equal than {} : P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(num_inst,
            meanpre * 100.0, meanrecall * 100.0, meanf1 * 100.0))
    return meanpre, meanrecall, meanf1

def calculate_hit_k(key, all_probs, k, relation_type, instance_number):
    hit_k = [[0 for _ in range(len(k))] for _ in range(len(relation_type))]
    maxkey = max(key)+1
    rel_num = [0 for _ in range(len(relation_type))]
    for rel in range(len(relation_type)):
        for row in range(len(key)):
            #print(maxkey, len(all_probs[row]))
            assert len(all_probs[row]) == maxkey
            if key[row] == relation_type[rel]:
                rel_num[rel]+=1
                prob_order = np.argsort(-np.asarray(all_probs[row]))
                for i in range(len(k)):
                    if key[row] in prob_order[:k[i]]:
                        hit_k[rel][i]+=1

    hit_k = np.asarray(hit_k)
    rel_num = np.asarray(rel_num)
    rel_num = (1/rel_num).reshape(-1,1)
    hit_k = hit_k*rel_num
    hit_k = np.mean(hit_k,axis=0)
    print("For the relations with number of instances <= ",instance_number)
    for i in range(hit_k.shape[0]):
        print('Hit@',k[i],"= {:.3%}".format(hit_k[i]))

def score(key, prediction, all_probs, DATASET, verbose=False, train_stat=None, relation_filter=None):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation    = Counter()
    train_gold_by_relation = Counter()

    label2id = constant.LABEL_TO_ID

    if DATASET == TACRED:
        NO_RELATION = 'no_relation'
    elif DATASET == NYT:
        NO_RELATION = 'NA'
    else:
#    elif DATASET == WEBNLG or DATASET == NYT24 or DATASET == NYT29:
        NO_RELATION = 'all_positive'

    # Loop over the data to compute a score
    for row in range(len(key)):
        gold = key[row]
        guess = prediction[row]
        if NO_RELATION == 'all_positive':
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1
        else:
            if gold == NO_RELATION and guess == NO_RELATION:
                pass
            elif gold == NO_RELATION and guess != NO_RELATION:
                guessed_by_relation[guess] += 1
            elif gold != NO_RELATION and guess == NO_RELATION:
                gold_by_relation[gold] += 1
            elif gold != NO_RELATION and guess != NO_RELATION:
                guessed_by_relation[guess] += 1
                gold_by_relation[gold] += 1
                if gold == guess:
                    correct_by_relation[guess] += 1
    gold_stat = train_stat if train_stat is not None else key
    for row in range(len(gold_stat)):
        gold = gold_stat[row]
        if NO_RELATION == 'all_positive':
            train_gold_by_relation[gold] += 1
        else:
            if gold != NO_RELATION:
                train_gold_by_relation[gold] += 1
    print(train_gold_by_relation)

    inst_no_5 = []
    inst_no_10 = []
    inst_no_20 = []
    inst_no_30 = []
    inst_no_100 = []
    inst_no_200 = []
    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            if relation_filter is not None and relation not in relation_filter:
                continue
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold    = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)

            if recall == 0:
                prec = 0
            if prec == 0:
                recall = 0

            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1: sys.stdout.write(' ')
            if prec < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1: sys.stdout.write(' ')
            if recall < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1: sys.stdout.write(' ')
            if f1 < 1.0: sys.stdout.write(' ')
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
            #### NOTE: changed
            gold = train_gold_by_relation[relation]
            if verbose:
                if DATASET in [NYT24, NYT, NYT29]:
                    if gold <= 100:
                        inst_no_20.append([recall, prec, f1])
                    elif gold <= 200:
                        inst_no_100.append([recall, prec, f1])
                    # if gold <= 250:
                    #     inst_no_20.append([recall, prec, f1])
                    # elif gold <= 1000:
                    #     inst_no_100.append([recall, prec, f1])
                    else:
                        inst_no_200.append([recall, prec, f1])
                else:
                    if gold <= 50:
                        inst_no_20.append([recall, prec, f1])
                    elif gold <= 100:
                        inst_no_100.append([recall, prec, f1])
                    else:
                        inst_no_200.append([recall, prec, f1])
            ###
            # if verbose and DATASET == TACRED:
            #     if gold <= 10:
            #         inst_no_5.append([recall, prec, f1])
            #     if gold <= 50:
            #         inst_no_10.append([recall, prec, f1])
            #     if gold <= 100:
            #         inst_no_20.append([recall, prec, f1])
            #     if gold > 100:
            #         inst_no_30.append([recall, prec, f1])
            # if verbose and (DATASET == NYT or DATASET == NYT24 or DATASET == NYT29 or DATASET == WEBNLG or DATASET ==FewRel or DATASET == Wiki80): #NYT
            #     if gold <= 20:
            #         inst_no_5.append([recall, prec, f1])
            #     if gold <= 50:
            #         inst_no_10.append([recall, prec, f1])
            #     if gold <= 100:
            #         inst_no_20.append([recall, prec, f1])
            #     if gold <= 400:
            #         inst_no_30.append([recall, prec, f1])
            #     if gold > 400:
            #         inst_no_200.append([recall, prec, f1])

        print("")

    if verbose:
        print("Num classes per split:", len(inst_no_20), len(inst_no_100), len(inst_no_200))
    # Print the aggregate score
    if verbose and (DATASET == TACRED or DATASET == WEBNLG) : # TACRED, WEBNLG
        print("Trigger Classification for Mean recall, Mean pre and Mean f1")

        calc_mean(inst_no_20,20)
        calc_mean(inst_no_100,100)
        calc_mean(inst_no_200,200)
        # calc_mean(inst_no_30,100, True)
        print("Final Score:")

    if verbose and (DATASET == NYT or DATASET == NYT24 or DATASET == NYT29 or DATASET == WEBNLG or DATASET ==FewRel or DATASET == Wiki80): # NYT
        print("Trigger Classification for Mean recall, Mean pre and Mean f1")

        calc_mean(inst_no_20,100)
        calc_mean(inst_no_100,200)
        calc_mean(inst_no_200,1000)
        print("Final Score:")


    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro   = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    print( "Precision (micro): {:.3%}".format(prec_micro) )
    print( "   Recall (micro): {:.3%}".format(recall_micro) )
    print( "       F1 (micro): {:.3%}".format(f1_micro) )

    label2id = constant.LABEL_TO_ID

    if all_probs!=None:
        key_id = [label2id[p] for p in key]
        if DATASET == TACRED:
            instance_10 = []
            instance_50 = []
            instance_100 = []
            for rel in gold_by_relation.keys():
                if gold_by_relation[rel] <= 10:
                    instance_10.append(label2id[rel])
                if gold_by_relation[rel] <=50:
                    instance_50.append(label2id[rel])
                if gold_by_relation[rel] <= 100:
                    instance_100.append(label2id[rel])
            calculate_hit_k(key_id, all_probs, [3,5,10,20], instance_10, 10)
            calculate_hit_k(key_id, all_probs, [3,5,10,20], instance_50, 50)
            calculate_hit_k(key_id, all_probs, [3,5,10,20], instance_100, 100)
        elif DATASET == NYT or DATASET == NYT24 or DATASET == NYT29 or DATASET == WEBNLG:
            instance_20 = []
            instance_50 = []
            instance_100 = []
            instance_400 = []
            for rel in gold_by_relation.keys():
                if gold_by_relation[rel] <= 20:
                    instance_20.append(label2id[rel])
                if gold_by_relation[rel] <=50:
                    instance_50.append(label2id[rel])
                if gold_by_relation[rel] <= 100:
                    instance_100.append(label2id[rel])
                if gold_by_relation[rel] <= 400:
                    instance_400.append(label2id[rel])
            calculate_hit_k(key_id, all_probs, [3,5,10,20], instance_20, 20)
            calculate_hit_k(key_id, all_probs, [3,5,10,20], instance_50, 50)
            calculate_hit_k(key_id, all_probs, [3,5,10,20], instance_100, 100)
            calculate_hit_k(key_id, all_probs, [3,5,10,20], instance_400, 400)

    return prec_micro, recall_micro, f1_micro

if __name__ == "__main__":
    # Parse the arguments from stdin
    args = parse_arguments()
    key = [str(line).rstrip('\n') for line in open(str(args.gold_file))]
    prediction = [str(line).rstrip('\n') for line in open(str(args.pred_file))]

    # Check that the lengths match
    if len(prediction) != len(key):
        print("Gold and prediction file must have same number of elements: %d in gold vs %d in prediction" % (len(key), len(prediction)))
        exit(1)
    
    # Score the predictions
    score(key, prediction, verbose=True)

