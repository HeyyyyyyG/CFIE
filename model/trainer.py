"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.gcn import GCNClassifier
from model.bert import BERTRelationModel
from utils import constant, torch_utils

from transformers import AdamW, get_linear_schedule_with_warmup

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
            if not self.opt['cuda']:
                checkpoint = torch.load(filename, map_location='cpu')
            else:
                checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        ## baseline
        if self.opt.get('tau', False):
            model_dict = self.model.state_dict()
            for key in model_dict.keys():
                if 'tau' in key and key not in checkpoint['model'].keys():
                    checkpoint['model'][key] = model_dict[key]
                elif 'tau' in key and key in checkpoint['model'].keys():
                    print("tau:", checkpoint['model'][key])
            self.model.load_state_dict(checkpoint['model'])
        else:
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

    ### baseline
    def prepare_crt(self):
        self.decoupled_freeze()
        # self.reset_classifiers()

    def prepare_lws(self):
        self.tau_freeze()

    def tau_freeze(self):
        # freeze all parameters except tau learning param
        for param_name, param in self.model.named_parameters():
            if 'tau' not in param_name:
                param.requires_grad = False
        
    def decoupled_freeze(self):
        # freeze all parameters except the classifiers
        for param_name, param in self.model.named_parameters():
            if 'classifier' not in param_name:
                param.requires_grad = False
            # print('  | ', param_name, param.requires_grad)

    def reset_classifiers(self):
        for i in self.model.classifiers:
            i.reset_parameters()


def unpack_batch(batch, cuda, mr=False):
    if cuda:
        inputs = [Variable(b.cuda()) for b in batch[:10]]
        labels = Variable(batch[10].cuda())
    else:
        inputs = [Variable(b) for b in batch[:10]]
        labels = Variable(batch[10])
    tokens = batch[0]
    head = batch[5]
    subj_pos = batch[6]
    obj_pos = batch[7]
    lens = batch[1].eq(0).long().sum(1).squeeze()

    if mr:
        rel_mask = batch[11].cuda() if cuda else batch[11]
        rel_num = batch[12]
        return inputs, labels, tokens, head, subj_pos, obj_pos, lens, rel_mask, rel_num  # , tokens_flatten
    else:

        return inputs, labels, tokens, head, subj_pos, obj_pos, lens

class GCNTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.model = GCNClassifier(opt, emb_matrix=emb_matrix)

        ## baseline
        loss = opt.get('loss', 'ce')
        print("Loss function:", loss)
        if loss == 'focal':
            from utils.loss import FocalLoss
            self.criterion = FocalLoss()
        elif loss == 'dice':
            from utils.loss import DiceLoss
            self.criterion = DiceLoss()
        elif loss == 'lifted':
            from pytorch_metric_learning import losses
            self.criterion = losses.LiftedStructureLoss()
        elif loss == 'ce':
            if opt['mr']:
                self.criterion = nn.BCEWithLogitsLoss(reduction='none')
            else:
                self.criterion = nn.CrossEntropyLoss()
        else:
            raise Exception("Unrecognized loss type:", loss)

        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])
        self.effect_type = opt['effect_type']

    def update(self, batch, eval):

        if self.opt['mr']:
            inputs, labels, tokens, head, subj_pos, obj_pos, lens, rel_mask, rel_num = unpack_batch(batch, self.opt['cuda'], mr=True)
        else:
            inputs, labels, tokens, head, subj_pos, obj_pos, lens = unpack_batch(batch, self.opt['cuda'], self.opt['mr'])

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        main_logits, logtis_dict, original_logits, intervention_logits, pooling_output = self.model(inputs, eval)

        if self.opt['mr']:
            # loss = sum([self.criterion(logits[:,r_no], labels[:,r_no]) for r_no in range(max_rel_no)])/max_rel_no
            loss = torch.sum(self.criterion(main_logits, labels.float()) * rel_mask.unsqueeze(2)) / torch.sum(rel_mask)

        else:
            loss = self.criterion(main_logits, labels)

        if self.opt['ner_loss']:
            loss = loss + ner_loss

        if loss.item() > 100:
            print("loss is abnormal", loss.item())

        loss_weight = 0.9

        if self.effect_type != 'None':
            if self.opt['mr']:
                loss_i2r = torch.sum(self.criterion(logtis_dict['i2rlogits'], labels.float()) * rel_mask.unsqueeze(2)) / torch.sum(rel_mask)
                loss_z2r = torch.sum(self.criterion(logtis_dict['z2rlogits'], labels.float()) * rel_mask.unsqueeze(2)) / torch.sum(rel_mask)
                loss_x2r = torch.sum(self.criterion(logtis_dict['x2rlogits'], labels.float()) * rel_mask.unsqueeze(2)) / torch.sum(rel_mask)
            else:
                loss_i2r = self.criterion(logtis_dict['i2rlogits'], labels)
                loss_z2r = self.criterion(logtis_dict['z2rlogits'], labels)
                loss_x2r = self.criterion(logtis_dict['x2rlogits'], labels)
            # loss = loss + loss_weight * loss_i2r + loss_weight * loss_z2r  + loss_x2r
            if self.opt['cfgen']:
                cflogits = logtis_dict['cflogits']
                cflogits = F.softmax(cflogits, dim=1)
                ### MSE start
                # negative_labels = torch.zeros_like(cflogits)
                # ones = torch.ones_like(labels).type_as(negative_labels).unsqueeze(1)
                # label_T = labels.unsqueeze(1)
                # negative_labels.scatter_(1, label_T, ones)
                # intervention_loss = negative_labels*cflogits
                # loss_inter = (intervention_loss**2).sum()
                ### MSE end

                ### Negative loss
                loss_inter = self.criterion(cflogits, labels)
                loss = loss + loss_weight * loss_i2r + loss_weight * loss_z2r + loss_x2r - loss_inter

                ### Positive anti-loss
                # cflogits = 1-cflogits
                # loss_inter = self.criterion(cflogits, labels)
                # loss = loss + loss_weight * loss_i2r + loss_weight * loss_z2r + loss_x2r + loss_inter

                # loss = self.model.weighted_loss(loss, loss_i2r, loss_z2r, loss_x2r)
            else:
                # loss = self.model.weighted_loss(loss, loss_i2r, loss_z2r, loss_x2r)
                loss = loss + loss_weight * loss_i2r + loss_weight * loss_x2r

        # l2 decay on all conv layers
        if self.opt.get('conv_l2', 0) > 0:
            loss += self.model.conv_l2() * self.opt['conv_l2']
        # l2 penalty on output representations
        if self.opt.get('pooling_l2', 0) > 0:
            loss += self.opt['pooling_l2'] * (pooling_output ** 2).sum(1).mean()
        loss_val = loss.item()
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, eval, unsort=True):
        if self.opt['mr']:
            inputs, labels, tokens, head, subj_pos, obj_pos, lens, rel_mask, rel_num = unpack_batch(batch, self.opt['cuda'], mr=True)
        else:
            inputs, labels, tokens, head, subj_pos, obj_pos, lens = unpack_batch(batch, self.opt['cuda'])

        orig_idx = batch[-1]
        # forward
        self.model.eval()

        logits, logtis_dict, original_logits, intervention_logits, pooling_output  = self.model(inputs, eval)

        if self.opt['mr']:
            # loss = sum([self.criterion(logits[:,r_no], labels[:,r_no]) for r_no in range(max_rel_no)])/max_rel_no
            loss = torch.sum(self.criterion(logits, labels.float()) * rel_mask.unsqueeze(2)) / torch.sum(rel_mask)

        else:
            loss = self.criterion(logits, labels)

        if self.opt['ner_loss']:
            loss = loss + ner_loss

        loss_weight = 0.9

        if self.effect_type != 'None':
            if self.opt['mr']:
                loss_i2r = torch.sum(self.criterion(logtis_dict['i2rlogits'], labels.float()) * rel_mask.unsqueeze(2)) / torch.sum(rel_mask)
                loss_z2r = torch.sum(self.criterion(logtis_dict['z2rlogits'], labels.float()) * rel_mask.unsqueeze(2)) / torch.sum(rel_mask)
                loss_x2r = torch.sum(self.criterion(logtis_dict['x2rlogits'], labels.float()) * rel_mask.unsqueeze(2)) / torch.sum(rel_mask)
            else:
                loss_i2r = self.criterion(logtis_dict['i2rlogits'], labels)
                loss_z2r = self.criterion(logtis_dict['z2rlogits'], labels)
                loss_x2r = self.criterion(logtis_dict['x2rlogits'], labels)
            loss = loss + loss_weight * loss_i2r + loss_weight * loss_z2r  + loss_x2r

        if self.opt['mr']:
            probs = F.softmax(logits, 2).data.cpu().numpy().tolist()
            predictions = np.argmax(logits.data.cpu().numpy(), axis=2).tolist()
            if unsort:
                _, predictions, probs, rel_num = [list(t) for t in zip(*sorted(zip(orig_idx,predictions, probs, rel_num)))]

        else:
            probs = F.softmax(logits, 1).data.cpu().numpy().tolist()
            predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
            if unsort:
                _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                        predictions, probs)))]


        if self.effect_type != 'None':
            if self.opt['mr']:
                probs_orig = F.softmax(original_logits, 2).data.cpu().numpy().tolist()
                predictions_orig = np.argmax(original_logits.data.cpu().numpy(), axis=2).tolist()
                if unsort:
                    _, predictions_orig, probs_orig = [list(t) for t in
                                                      zip(*sorted(zip(orig_idx, predictions_orig, probs_orig)))]

                probs_interv = F.softmax(intervention_logits, 2).data.cpu().numpy().tolist()
                predictions_interv = np.argmax(intervention_logits.data.cpu().numpy(), axis=2).tolist()
                if unsort:
                    _, predictions_interv, probs_interv = [list(t) for t in zip(*sorted(zip(orig_idx, \
                                                                                        predictions_interv, probs_interv)))]
            else:
                probs_orig = F.softmax(original_logits, 1).data.cpu().numpy().tolist()
                predictions_orig = np.argmax(original_logits.data.cpu().numpy(), axis=1).tolist()
                if unsort:
                    _, predictions_orig, probs_orig = [list(t) for t in zip(*sorted(zip(orig_idx,\
                            predictions_orig, probs_orig)))]

                probs_interv = F.softmax(intervention_logits, 1).data.cpu().numpy().tolist()
                predictions_interv = np.argmax(intervention_logits.data.cpu().numpy(), axis=1).tolist()
                if unsort:
                    _, predictions_interv, probs_interv = [list(t) for t in zip(*sorted(zip(orig_idx,\
                            predictions_interv, probs_interv)))]

        else:
            predictions_orig = probs_orig = probs_interv = predictions_interv = None

        effect_score = 0
        if logtis_dict is not None:
            i2r_effect = logtis_dict['i2rlogits']
            x2r_effect = logtis_dict['x2rlogits']
            z2r_effect = logtis_dict['z2rlogits']
            effect_logits = logits.unsqueeze(1)
            effect_logits = torch.stack([i2r_effect, x2r_effect, z2r_effect], dim=1)
            # print(effect_logits.size())
            effect_score = self.compute_effect_score(effect_logits, labels)

        if self.opt['mr']:
            return predictions, probs, loss.item(), predictions_orig, probs_orig, predictions_interv, probs_interv, rel_num
        else:
            return predictions, probs, loss.item(), predictions_orig, probs_orig, predictions_interv, probs_interv, effect_score

    def compute_effect_score(self, logits, labels):
        logits = F.softmax(logits, -1)
        # logits = torch.abs(logits)
        effective_logits = torch.gather(logits, -1, labels.view(-1, 1, 1).repeat(1, 3, 1))
        effect_score = effective_logits.mean(dim=0)
        return effect_score


class GCNBertTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.model = BERTRelationModel(opt) 
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.negative_criterion = nn.MSELoss()
        # self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-3},
            {'params': [p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=opt['lr'])
        self.effect_type = opt['effect_type']

    def prepare_crt(self):
        self.model.decoupled_freeze()
        self.model.reset_classifiers()

    def unpack_batch(self, batch, cuda):
        if cuda:
            inputs = [Variable(b.cuda()) for b in batch[:10]]
            labels = Variable(batch[10].cuda())
        else:
            inputs = [Variable(b) for b in batch[:10]]
            labels = Variable(batch[10])
        tokens = batch[0]
        head = batch[5]
        subj_pos = batch[6]
        obj_pos = batch[7]
        lens = batch[1].eq(0).long().sum(1).squeeze()
        
        # for bert
        bert_inputs = batch[-1]
        for i in bert_inputs.keys():
            if cuda:
                try:
                    bert_inputs[i] = bert_inputs[i].cuda()
                except:
                    continue
        inputs[0] = bert_inputs
        return inputs, labels, tokens, head, subj_pos, obj_pos, lens

    def update(self, batch, eval):
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = self.unpack_batch(batch, self.opt['cuda'])

        # step forward
        self.model.train()
        self.optimizer.zero_grad()

        main_logits, logtis_dict, original_logits, intervention_logits, pooling_output = self.model(inputs, eval)
        loss = self.criterion(main_logits, labels)

        if self.opt['ner_loss']:
            loss = loss + ner_loss

        if loss.item() > 100:
            print("loss is abnormal")

        loss_weight = 0.9

        if self.effect_type != 'None':
            loss_i2r = self.criterion(logtis_dict['i2rlogits'], labels)
            loss_x2r = self.criterion(logtis_dict['x2rlogits'], labels)
            loss_z2r = self.criterion(logtis_dict['z2rlogits'], labels)
            if self.opt['cfgen']:
                cflogits = logtis_dict['cflogits']
                cflogits = F.softmax(cflogits, dim=1)
                # negative_labels = torch.zeros_like(cflogits)
                # ones = torch.ones_like(labels).type_as(negative_labels).unsqueeze(1)
                # label_T = labels.unsqueeze(1)
                # negative_labels.scatter_(1, label_T, ones)
                # intervention_loss = negative_labels*cflogits
                # loss_inter = (intervention_loss**2).sum()
                loss_inter = self.criterion(cflogits, labels)
                loss = loss + loss_weight * loss_i2r + loss_weight * loss_z2r + loss_x2r - loss_inter
            else:
                loss = loss + loss_weight * loss_i2r + loss_weight * loss_z2r + loss_x2r
            #loss = loss + loss_weight * loss_i2r + loss_weight * loss_x2r

        # l2 decay on all conv layers
        if self.opt.get('conv_l2', 0) > 0:
            loss += self.model.conv_l2() * self.opt['conv_l2']
        # l2 penalty on output representations
        if self.opt.get('pooling_l2', 0) > 0:
            loss += self.opt['pooling_l2'] * (pooling_output ** 2).sum(1).mean()
        loss_val = loss.item()
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, eval, unsort=True):
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = self.unpack_batch(batch, self.opt['cuda'])
        orig_idx = batch[11]
        # forward
        self.model.eval()


        logits, logtis_dict, original_logits, intervention_logits, pooling_output  = self.model(inputs, eval)

        loss = self.criterion(logits, labels)

        if self.opt['ner_loss']:
            loss = loss + ner_loss

        loss_weight = 0.9

        if self.effect_type != 'None':
            loss_i2r = self.criterion(logtis_dict['i2rlogits'], labels)
            loss_z2r = self.criterion(logtis_dict['z2rlogits'], labels)
            loss_x2r = self.criterion(logtis_dict['x2rlogits'], labels)
            loss = loss + loss_weight * loss_i2r + loss_weight * loss_z2r + loss_x2r
            #loss = loss + loss_weight * loss_i2r + loss_weight * loss_x2r

        probs = F.softmax(logits, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs)))]

        if self.effect_type != 'None':
            probs_orig = F.softmax(original_logits, 1).data.cpu().numpy().tolist()
            predictions_orig = np.argmax(original_logits.data.cpu().numpy(), axis=1).tolist()
            if unsort:
                _, predictions_orig, probs_orig = [list(t) for t in zip(*sorted(zip(orig_idx,\
                        predictions_orig, probs_orig)))]

            probs_interv = F.softmax(intervention_logits, 1).data.cpu().numpy().tolist()
            predictions_interv = np.argmax(intervention_logits.data.cpu().numpy(), axis=1).tolist()
            if unsort:
                _, predictions_orig, probs_orig = [list(t) for t in zip(*sorted(zip(orig_idx,\
                        predictions_interv, probs_interv)))]

        else:
            predictions_orig = probs_orig = probs_interv = predictions_interv = None

        if self.opt['mr']:
            return predictions, probs, loss.item(), predictions_orig, probs_orig, predictions_interv, probs_interv, rel_num
        else:
            return predictions, probs, loss.item(), predictions_orig, probs_orig, predictions_interv, probs_interv
