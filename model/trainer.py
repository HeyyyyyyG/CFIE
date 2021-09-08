"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


from model.gcn import NERClassifier
from utils import constant, torch_utils
from neuronlp2.nn import CharCNN



class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

   

class NERTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None, char_emb_matrix=None):
        self.opt = opt

        self.emb_matrix = emb_matrix
        self.char_emb_matrix = char_emb_matrix

        print("using Softmax classifier")
        self.model = NERClassifier(opt, emb_matrix=emb_matrix,char_emb_matrix=char_emb_matrix)
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        
        self.criterion = nn.CrossEntropyLoss()
        

        self.effect_type=opt['effect_type']
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def unpack_batch(self,batch, cuda):
        if cuda:
            inputs = [Variable(b.cuda()) for b in batch[:8]]
            labels = Variable(batch[9].cuda())
        else:
            inputs = [Variable(b) for b in batch[:8]]
            labels = Variable(batch[9])
        tokens = batch[0]
        mask = batch[1]
        chars_seq = batch[2]
        lens = batch[1].long().sum(1).squeeze()
        return inputs, tokens, mask, chars_seq, labels, lens


    def update(self, batch, eval):
        inputs, tokens, mask, chars_seq, labels, lens = self.unpack_batch(batch, self.opt['cuda'])
        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        loss_weight = 1

     
        logits, logit_dict= self.model(inputs,eval)
        loss = self.criterion(logits.transpose(1, 2), labels).requires_grad_()

        if self.effect_type != 'None':
            i2rlogits = logit_dict['i2rlogits']
            pos2rlogits = logit_dict['z2rlogits']

            x2rlogits = logit_dict['x2rlogits']
            loss = loss + loss_weight * self.criterion(i2rlogits.transpose(1, 2), labels) + loss_weight * self.criterion(pos2rlogits.transpose(1, 2), labels)  + loss_weight * self.criterion(x2rlogits.transpose(1, 2), labels)
        loss_val = loss.item()
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, eval, leading_symbolic=0):
        inputs, tokens, mask, chars_seq, labels, lens = self.unpack_batch(batch, self.opt['cuda'])
        # forward
        self.model.eval()
        
        logits,logit_dict = self.model(inputs, eval)
        logits = logits.transpose(1, 2)
        _, preds = torch.max(logits[:, leading_symbolic:], dim=1)
        preds += leading_symbolic
        if mask is not None:
            preds = preds.cpu() * mask.long()

        return preds


