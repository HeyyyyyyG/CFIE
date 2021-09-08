
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

from data.loader import NERDataLoader
from model.trainer import NERTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab

class Span:

    def __init__(self, left, right, type):
        self.left = left
        self.right = right
        self.type = type

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right and self.type == other.type

    def __hash__(self):
        return hash((self.left, self.right, self.type))

    def to_str(self, sent):
        return str(sent[self.left: (self.right+1)]) + ","+self.type

def evaluate_num(batch_pred_ids, batch_gold_ids, word_seq_lens, idx2label):
    """
    evaluate the batch of instances
    :param batch_insts:
    :param batch_pred_ids:
    :param batch_gold_ids:
    :param word_seq_lens:
    :param idx2label:
    :return:
    """
    p = 0
    total_entity = 0
    total_predict = 0
    word_seq_lens = word_seq_lens.tolist()
    key = set()
    predict=set()
    for idx in range(len(batch_pred_ids)):
        length = word_seq_lens[idx]
        output = batch_gold_ids[idx][:length].tolist()
        prediction = batch_pred_ids[idx][:length].tolist()
        output = [idx2label[l] for l in output]
        prediction =[idx2label[l] for l in prediction]
        #convert to span
        output_spans = set()
        start = -1
        for i in range(len(output)):
            if output[i].startswith("B-"):
                start = i
            if output[i].startswith("E-"):
                end = i
                output_spans.add(Span(start, end, output[i][2:]))
            if output[i].startswith("S-"):
                output_spans.add(Span(i, i, output[i][2:]))
        predict_spans = set()
        for i in range(len(prediction)):
            if prediction[i].startswith("B-"):
                start = i
            if prediction[i].startswith("E-"):
                end = i
                predict_spans.add(Span(start, end, prediction[i][2:]))
            if prediction[i].startswith("S-"):
                predict_spans.add(Span(i, i, prediction[i][2:]))

        total_entity += len(output_spans)
        total_predict += len(predict_spans)
        p += len(predict_spans.intersection(output_spans))
        key.update(output_spans)
        predict.update(predict_spans)

    return np.asarray([p, total_predict, total_entity], dtype=int),key, predict


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset/atis')
parser.add_argument('--vocab_dir', type=str, default='dataset/atis')
parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--char_emb_dim', type=int, default=100, help='Char embedding dimension.')
parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=200, help='RNN hidden state size.')
parser.add_argument('--num_gcn_layers', type=int, default=1, help='Num of GCN layers.')

parser.add_argument('--gcn_dropout', type=float, default=0.5, help='GCN layer dropout rate.')
parser.add_argument('--word_dropout', type=float, default=0.06, help='The rate at which randomly set a word to UNK.')

parser.set_defaults(lower=False)

parser.add_argument('--rnn_hidden', type=int, default=200, help='RNN hidden state size.')
parser.add_argument('--rnn_layers', type=int, default=1, help='Number of RNN layers.')

parser.add_argument('--lr', type=float, default=0.001, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam', help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=1000, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=64, help='Training batch size.')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=1, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

parser.add_argument('--load', dest='load', action='store_true', help='Load pretrained model.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')

parser.add_argument('--effect_type', type=str, default="None", help='[None, CAUSAL]')

parser.add_argument('--alpha', type=float, default=1.2, help='alpha')
parser.add_argument('--max_len', type=int, default=300, help='max seq length')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
    
elif args.cuda:
    torch.cuda.manual_seed(args.seed)
    
init_time = time.time()

# make opt
opt = vars(args)
label2id = constant.LABEL_TO_ID
opt['num_class'] = len(label2id)
print(opt)
# load vocab
vocab_file = opt['vocab_dir'] + '/vocab.pkl'
char_vocab_file = opt['vocab_dir'] + '/charvocab.pkl'
vocab = Vocab(vocab_file, load=True)
char_vocab = Vocab(char_vocab_file, load=True)
opt['vocab_size'] = vocab.size
opt['char_vocab_size'] = char_vocab.size
emb_file = opt['vocab_dir'] + '/embedding.npy'
emb_matrix = np.load(emb_file)
char_emb_file = opt['vocab_dir'] + '/charembedding.npy'
char_emb_matrix = np.load(char_emb_file)
print('char_emb_file',char_emb_file)
print('char_emb_matrix.shape',char_emb_matrix.shape)
assert emb_matrix.shape[0] == vocab.size
assert emb_matrix.shape[1] == opt['emb_dim']


# load data
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))

train_batch = NERDataLoader(opt['data_dir'] + '/train.json', opt['batch_size'], opt, vocab, char_vocab, evaluation=False)
dev_batch = NERDataLoader(opt['data_dir'] + '/dev.json', opt['batch_size'], opt, vocab, char_vocab, evaluation=True)


model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)

# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
vocab.save(model_save_dir + '/vocab.pkl')
char_vocab.save(model_save_dir + '/charvocab.pkl')
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'])

# print model info
helper.print_config(opt)

# model
if not opt['load']:
    trainer = NERTrainer(opt, emb_matrix=emb_matrix, char_emb_matrix = char_emb_matrix)

else:
    # load pretrained model
    model_file = opt['model_file'] 
    print("Loading model from {}".format(model_file))
    model_opt = torch_utils.load_config(model_file)
    model_opt['optim'] = opt['optim']

    trainer = NERTrainer(model_opt,emb_matrix=emb_matrix, char_emb_matrix = char_emb_matrix)
    trainer.load(model_file)   


id2label = dict([(v,k) for k,v in label2id.items()])
dev_score_history = []

global_step = 0
global_start_time = time.time()
max_steps = len(train_batch) * opt['num_epoch']

# start training
current_lr = opt['lr']
trainer.update_lr(current_lr)
best_f = 0
for epoch in range(1, opt['num_epoch']+1):
    print("epoch {} running...".format(epoch))
    print("learning rate is: {}".format(trainer.optimizer.state_dict()['param_groups'][0]['lr']))
    train_loss = 0
    for i, batch in enumerate(train_batch):
        start_time = time.time()
        global_step += 1
        loss = trainer.update(batch,eval=False)
        train_loss += loss
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
    print("Evaluating on dev set...")
    f_p = 0
    f_total_predict = 0
    f_total_entity = 0
    key=set()
    prediction=set()
    for i, batch in enumerate(dev_batch):
        preds = trainer.predict(batch,eval=True)

        array, output_spans, predict_spans = evaluate_num(preds, batch[9], batch[1].long().sum(1).squeeze(), id2label)
        p=array[0]
        total_predict=array[1]
        total_entity=array[2]
        key.update(output_spans)
        prediction.update(predict_spans)
        
    dev_p, dev_r, dev_f1 = scorer.score_span(key, prediction,verbose=False)
    
    # save
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    trainer.save(model_file, epoch)
    best_model_file_name = '/best_model.pt'

    if epoch == 1 or dev_f1 > best_f:
        copyfile(model_file, model_save_dir + '/' + best_model_file_name)
        print("new best model saved.")
        file_logger.log("new best model saved at epoch {}: {:.2f}\t{:.2f}\t{:.2f}"\
            .format(epoch, dev_p*100, dev_r*100, dev_f1*100))
        os.remove(model_file)
        
    # change lr
    if dev_f1 > best_f:
        best_f = dev_f1
        if epoch > 3:
            current_lr *= opt['lr_decay']
            print('change lr to {:.3f}'.format(current_lr))
            trainer.update_lr(current_lr)
    print("epoch {}: train_loss = {:.6f}, dev_f1 = {:.4f}, dev_p ={:.4f}, dev_r ={:.4f}, best_f = {:.4f} ".format(epoch,train_loss, dev_f1, dev_p, dev_r, best_f ))
    
    

