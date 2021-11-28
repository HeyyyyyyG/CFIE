"""
Train a model on TACRED.
"""

import os
import sys
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from data.loader import DataLoader, BertDataLoader
from model.trainer import GCNTrainer, GCNBertTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab
from sklearn.metrics import f1_score, recall_score, precision_score

TACRED = 'tacred'
NYT = 'nyt'
NYT24 = 'nyt24'
NYT29 = 'nyt29'
ACE = 'ace2005'
WEBNLG = 'webnlg'
FewRel = 'fewrel'
Wiki80 = 'wiki80'

'''
fuse_type == SUM
01/02 TACRED NONE/TDE
03/04 NYT24 NONE/TDE
05/06 NYT29 NONE/TDE
07/08 WEBNLG NONE/TDE. For 08/, need to comment self.gate2rel and opt['fuse_type'] as these modules are not implemented during training the model.

Fuse_type == GATED
09/10 NYT24 NONE/TDE. 
11/12 NYT29 NONE/TDE

15/16 NYT24 NONE/TDE multi-interventions

'''

# DATASET = TACRED
# DATASET = WEBNLG
DATASET = NYT24
# DATASET = NYT

parser = argparse.ArgumentParser()
parser.add_argument('--mr', dest='mr', action='store_false', help="multi relations for an instance")
parser.add_argument('--mi', dest='mi', action='store_false', help="multi interventions for a causal graph")
parser.set_defaults(mr=False)
parser.set_defaults(mi=False)

if DATASET == TACRED:
    parser.add_argument('--data_dir', type=str, default='dataset/tacred')
    parser.add_argument('--vocab_dir', type=str, default='dataset/tacred')
    parser.add_argument('--log_step', type=int, default=200, help='Print log every k steps.')

elif DATASET == NYT:
    parser.add_argument('--data_dir', type=str, default='dataset/nyt')
    parser.add_argument('--vocab_dir', type=str, default='dataset/nyt')
    parser.add_argument('--log_step', type=int, default=1000, help='Print log every k steps.')

elif DATASET == NYT24:
    parser.add_argument('--data_dir', type=str, default='dataset/nyt24')
    parser.add_argument('--vocab_dir', type=str, default='dataset/nyt24')
    parser.add_argument('--log_step', type=int, default=200, help='Print log every k steps.')
    # parser.set_defaults(mr=True)
    parser.set_defaults(mr=False)
    parser.set_defaults(mi=False)

elif DATASET == NYT29:
    parser.add_argument('--data_dir', type=str, default='dataset/nyt29')
    parser.add_argument('--vocab_dir', type=str, default='dataset/nyt29')
    parser.add_argument('--log_step', type=int, default=200, help='Print log every k steps.')
    parser.set_defaults(mr=True)
    parser.set_defaults(mi=False)

elif DATASET == ACE:
    parser.add_argument('--data_dir', type=str, default='dataset/ace2005_re')
    parser.add_argument('--vocab_dir', type=str, default='dataset/ace2005_re')
    parser.add_argument('--log_step', type=int, default=1000, help='Print log every k steps.')
    parser.set_defaults(mr=False)
    parser.set_defaults(mi=False)

elif DATASET == WEBNLG:
    parser.add_argument('--data_dir', type=str, default='dataset/webnlg')
    parser.add_argument('--vocab_dir', type=str, default='dataset/webnlg')
    parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
    parser.set_defaults(mr=True)
    # parser.set_defaults(mr=False)
    parser.set_defaults(mi=False)

elif DATASET == FewRel:
    parser.add_argument('--data_dir', type=str, default='dataset/fewrel')
    parser.add_argument('--vocab_dir', type=str, default='dataset/fewrel')
    parser.add_argument('--log_step', type=int, default=200, help='Print log every k steps.')
    parser.set_defaults(mr=False)
    parser.set_defaults(mi=False)

elif DATASET == Wiki80:
    parser.add_argument('--data_dir', type=str, default='dataset/wiki80')
    parser.add_argument('--vocab_dir', type=str, default='dataset/wiki80')
    parser.add_argument('--log_step', type=int, default=200, help='Print log every k steps.')
    parser.set_defaults(mr=False)
    parser.set_defaults(mi=False)

parser.add_argument('--dataset', type=str, default=DATASET, help='for computing Precision and recall.')
parser.add_argument('--effect_type', type=str, default="None", help='[TDE, NIE, TE, None]')
parser.add_argument('--fuse_type', type=str, default="SUM", help='[SUM, GATED]')

parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=200, help='RNN hidden state size.')
parser.add_argument('--num_layers', type=int, default=2, help='Num of RNN layers.')
parser.add_argument('--input_dropout', type=float, default=0.5, help='Input dropout rate.')
parser.add_argument('--gcn_dropout', type=float, default=0.5, help='GCN layer dropout rate.')
parser.add_argument('--word_dropout', type=float, default=0.04, help='The rate at which randomly set a word to UNK.')
parser.add_argument('--topn', type=int, default=1e10, help='Only finetune top N word embeddings.')
parser.add_argument('--max_len', type=int, default=300, help='RNN hidden state size.')
parser.add_argument('--lower', dest='lower', action='store_true', help='Lowercase all words.')
parser.add_argument('--no-lower', dest='lower', action='store_false')
parser.add_argument('--ner_loss', dest='ner_loss', action='store_false')
parser.add_argument('--ctx_I', dest='ctx_I', action='store_false')
parser.set_defaults(lower=False)
parser.set_defaults(ner_loss=False)
parser.set_defaults(ctx_I=True)

parser.add_argument('--prune_k', default=-1, type=int, help='Prune the dependency tree to <= K distance off the dependency path; set to -1 for no pruning.')
parser.add_argument('--conv_l2', type=float, default=0, help='L2-weight decay on conv layers only.')
parser.add_argument('--pooling', choices=['max', 'avg', 'sum'], default='max', help='Pooling function type. Default max.')
parser.add_argument('--pooling_l2', type=float, default=0, help='L2-penalty for all pooling output.')
parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')
parser.add_argument('--no_adj', dest='no_adj', action='store_true', help="Zero out adjacency matrix for ablation.")

parser.add_argument('--no-rnn', dest='rnn', action='store_false', help='Do not use RNN layer.')
parser.add_argument('--rnn_hidden', type=int, default=200, help='RNN hidden state size.')
parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')
parser.add_argument('--rnn_dropout', type=float, default=0.5, help='RNN dropout rate.')

parser.add_argument('--lr', type=float, default=1.0, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.98, help='Learning rate decay rate.')
parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='sgd', help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=100, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=50, help='Training batch size.')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
#parser.add_argument('--log_step', type=int, default=200, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

parser.add_argument('--load', dest='load', action='store_true', help='Load pretrained model.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')
parser.add_argument('--metric', type=str, choices=['f', 'p', 'r'], default='f', help='best model picking metric')

# bert args
parser.add_argument('--bert_model', type=str, default='bert-base-uncased', help='bert model name')
parser.add_argument('--multi_piece', type=str, default='first', choices=['first', 'average'])
parser.add_argument('--use_bert', default=False, action='store_true')
parser.add_argument('--freeze', default=False, action='store_true')

## baseline
parser.add_argument('--crt', default=False, action='store_true')
parser.add_argument('--lws', default=False, action='store_true')
parser.add_argument('--cbs', default=False, action='store_true')
parser.add_argument('--tau', type=float, default=0.5, help='tau normalization')
parser.add_argument('--loss', type=str, choices=['ce', 'focal', 'dice', 'lifted'], default='ce')
parser.add_argument('--no_gcn', default=False, action='store_true')

### aggcn baseline
parser.add_argument('--aggcn', default=False, action='store_true')
parser.add_argument('--heads', type=int, default=4, help='Num of heads in multi-head attention.')
parser.add_argument('--sublayer_first', type=int, default=2, help='Num of the first sublayers in dcgcn block.')
parser.add_argument('--sublayer_second', type=int, default=4, help='Num of the second sublayers in dcgcn block.')

# counterfactuals
parser.add_argument('--cfgen', default=False, action='store_true')
parser.add_argument('--context', default=False, action='store_true')
parser.add_argument('--dependency', action='store_true')
parser.add_argument('--instance_class', type=int, default=None)
parser.add_argument('--macro', default=False, action='store_true')

args = parser.parse_args()
USE_BERT = args.use_bert

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)
init_time = time.time()

# check dataset
# if args.dataset == TACRED:
#     args.data_dir='dataset/tacred'
#     args.vocab_dir='dataset/tacred'
#     args.log_step = 1000
# elif args.dataset == NYT:
#     args.data_dir='dataset/nyt_sr'
#     args.vocab_dir='dataset/nyt_sr'
#     args.log_step = 1000
# elif args.dataset == ACE:
#     args.data_dir='dataset/ace2005_re'
#     args.vocab_dir='dataset/ace2005_re'
#     args.log_step = 1000
# elif args.dataset == WEBNLG:
#     args.data_dir='dataset/webnlg_sr'
#     args.vocab_dir='dataset/webnlg_sr'
#     args.log_step = 1000
# else:
#     raise Exception("Unknown dataset")

# make opt
opt = vars(args)
label2id = constant.LABEL_TO_ID
opt['num_class'] = len(label2id)

# load vocab
vocab_file = opt['vocab_dir'] + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
opt['vocab_size'] = vocab.size
emb_file = opt['vocab_dir'] + '/embedding.npy'
emb_matrix = np.load(emb_file)
assert emb_matrix.shape[0] == vocab.size
assert emb_matrix.shape[1] == opt['emb_dim']

# load data
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
if USE_BERT:
    train_batch = BertDataLoader(opt['data_dir'] + '/train.json', opt['batch_size'], opt, vocab, evaluation=False, mr = opt['mr'])
    dev_batch = BertDataLoader(opt['data_dir'] + '/dev.json', opt['batch_size'], opt, vocab, evaluation=True, mr = opt['mr'])
else:
    train_batch = DataLoader(opt['data_dir'] + '/train.json', opt['batch_size'], opt, vocab, evaluation=False, mr = opt['mr'])
    dev_batch = DataLoader(opt['data_dir'] + '/dev.json', opt['batch_size'], opt, vocab, evaluation=True, mr = opt['mr'])

model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)

# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
vocab.save(model_save_dir + '/vocab.pkl')
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")
if opt['effect_type'] != "None":
    file_logger_orig = helper.FileLogger(model_save_dir + '/' + opt['log'] +"_orig", header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")
    file_logger_interv = helper.FileLogger(model_save_dir + '/' + opt['log'] +"_interv", header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")

# print model info
helper.print_config(opt)

# model
if not opt['load']:
    if opt['crt'] or opt['lws']:
        raise Exception("classifier retraining with no model loaded")
    if USE_BERT:
        trainer = GCNBertTrainer(opt, emb_matrix=emb_matrix)
    else:
        trainer = GCNTrainer(opt, emb_matrix=emb_matrix)
else:
    # load pretrained model
    model_file = opt['model_file']
    print("Loading model from {}".format(model_file))
    model_opt = torch_utils.load_config(model_file)
    model_opt['optim'] = opt['optim']
    if USE_BERT:
        trainer = GCNBertTrainer(opt, emb_matrix=emb_matrix)
    else:
        trainer = GCNTrainer(opt, emb_matrix=emb_matrix)
    trainer.load(model_file)
    ## baseline
    torch.autograd.set_detect_anomaly(True)
    if opt['crt']:
        trainer.prepare_crt()
    elif opt['lws']:
        trainer.prepare_lws()
        opt['use_tau'] = True

id2label = dict([(v,k) for k,v in label2id.items()])
dev_score_history = []
dev_score_history_o = []
dev_score_history_i = []

current_lr = opt['lr']

global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']

helper.save_config(opt, model_save_dir + '/config.json', verbose=True)

# start training
max_dev_f1 = 0
for epoch in range(1, opt['num_epoch']+1):
    train_loss = 0
    for i, batch in enumerate(train_batch):
        # if i > 10:
        #     break
        start_time = time.time()
        global_step += 1
        loss = trainer.update(batch, eval=False)
        train_loss += loss
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch,\
                    opt['num_epoch'], loss, duration, current_lr))

    # eval on dev
    print("Evaluating on dev set...")
    predictions = []
    predictions_orig = []
    predictions_interv = []
    effect_scores = 0
    dev_loss = 0
    for i, batch in enumerate(dev_batch):
        if opt['mr']:
            preds, _, loss, preds_orig, probs_orig, preds_interv, probs_interv, rel_nums = trainer.predict(batch, eval=True)
            batch_preds = []
            if opt['effect_type'] != "None":
                batch_preds_o = []
                batch_preds_i = []
                for p, p_o, p_i, rel in zip(preds, preds_orig, preds_interv, rel_nums):
                    batch_preds += p[:rel]
                    batch_preds_o += p_o[:rel]
                    batch_preds_i += p_i[:rel]
            else:
                for p, rel in zip(preds, rel_nums):
                    batch_preds += p[:rel]

            assert len(batch_preds) == sum(rel_nums)
            predictions += batch_preds
            if opt['effect_type'] != "None":
                predictions_orig += batch_preds_o
                predictions_interv += batch_preds_i

        else:
            preds, _, loss, preds_orig, probs_orig, preds_interv, probs_interv, batch_effect_score = trainer.predict(batch, eval=True)
            predictions += preds
            if opt['effect_type'] != "None":
                predictions_orig += preds_orig
                predictions_interv += preds_interv
                effect_scores += batch_effect_score
        dev_loss += loss
    effect_scores /= len(dev_batch)
    print("Final score:", effect_scores)
    predictions = [id2label[p] for p in predictions]

    if opt['effect_type'] != "None":
        predictions_orig = [id2label[p] for p in predictions_orig]
        predictions_interv = [id2label[p] for p in predictions_interv]

    train_loss = train_loss / train_batch.num_examples * opt['batch_size'] # avg loss per batch
    dev_loss = dev_loss / dev_batch.num_examples * opt['batch_size']

    dev_p, dev_r, dev_f1 = scorer.score(dev_batch.gold(), predictions, None, opt['dataset'], verbose=False)

    if opt['effect_type'] != "None":
        print("The results for (I+Z+X)->Y:")
        dev_p_o, dev_r_o, dev_f1_o = scorer.score(dev_batch.gold(), predictions_orig, None, opt['dataset'], verbose=False)
        print("The results for X_{i}->Y") # interverntion
        dev_p_i, dev_r_i, dev_f1_i = scorer.score(dev_batch.gold(), predictions_interv, None, opt['dataset'], verbose=False)

    print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".format(epoch,\
        train_loss, dev_loss, dev_f1))

    if opt['macro']:
        dev_f1 = f1_score(dev_batch.gold(), predictions, average='macro')
        dev_r = recall_score(dev_batch.gold(), predictions, average='macro')
        dev_p = precision_score(dev_batch.gold(), predictions, average='macro')

    ###NOTE: changed
    if opt['metric'] == 'f':
        dev_score = dev_f1
    elif opt['metric'] == 'r':
        dev_score = dev_r
    elif opt['metric'] == 'p':
        dev_score = dev_p
    else:
        raise Exception("Unsupported model picking metric, please choose from [f, r, p]")

    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, dev_loss, dev_score, max([dev_score] + dev_score_history)))

    if opt['effect_type'] != "None":
        file_logger_orig.log("{}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, 0, 0, dev_f1_o,
                                                                    max([dev_f1_o] + dev_score_history_o)))
        file_logger_interv.log("{}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, 0, 0, dev_f1_i, max([dev_f1_i] + dev_score_history_i)))

    # save
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    trainer.save(model_file, epoch)
    best_model_file_name = 'best_model.pt'
    if opt['crt']:
        best_model_file_name = 'best_retrain_model.pt'
    elif opt['lws']:
        best_model_file_name = 'best_lws_model.pt'
    if epoch == 1 or dev_score > max(dev_score_history):
        copyfile(model_file, model_save_dir + '/' + best_model_file_name)
        print("new best model saved.")
        file_logger.log("new best model saved at epoch {}: {:.2f}\t{:.2f}\t{:.2f}"\
            .format(epoch, dev_p*100, dev_r*100, dev_score*100))
    if epoch % opt['save_epoch'] != 0:
        os.remove(model_file)

    # lr schedule
    if len(dev_score_history) > opt['decay_epoch'] and dev_score <= dev_score_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad', 'adadelta']:
        current_lr *= opt['lr_decay']
        trainer.update_lr(current_lr)

    dev_score_history += [dev_score]
    print("")

print("Training ended with {} epochs.".format(epoch))

