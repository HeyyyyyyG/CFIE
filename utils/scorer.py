

import argparse
import sys
from collections import Counter

NO_RELATION = "O"
from utils import constant
train_sample = constant.train_sample
def parse_arguments():
    parser = argparse.ArgumentParser(description='Score a prediction file using the gold labels.')
    parser.add_argument('gold_file', help='The gold relation file; one relation per line')
    parser.add_argument('pred_file', help='A prediction file; one relation per line, in the same order as the gold file.')
    args = parser.parse_args()
    return args


def score_span(key, prediction, verbose=False):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation    = Counter()

    intersect = prediction.intersection(key)
    for sp in intersect:
        correct_by_relation[sp.type]+=1
    key = list(key)
    prediction = list(prediction)
    relation_type = []
    for sp in key:
        if sp.type not in relation_type:
            relation_type.append(sp.type)
    print('relation_type',relation_type)
    for rel in relation_type:
        for sp in key:
            if sp.type==rel:
                gold_by_relation[rel]+=1
        for sp in prediction:
            if sp.type==rel:
                guessed_by_relation[rel]+=1

    # Loop over the data to compute a score
    print(gold_by_relation)
    print(guessed_by_relation)
    print(correct_by_relation)

    # Print verbose information
    if verbose:
        print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
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
            sys.stdout.write("  #: %d" % train_sample[relation])
            sys.stdout.write("\n")
        print("")


    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro   = float(sum(correct_by_relation.values())) / float(sum(guessed_by_relation.values()))
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(sum(gold_by_relation.values()))
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)

    return prec_micro, recall_micro, f1_micro



