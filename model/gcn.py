"""
GCN model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.tree import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils
from model.counterfactual import get_shortest_path_tmp, get_neighbor

class GCNClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.gcn_model = GCNRelationModel(opt, emb_matrix=emb_matrix)
        self.opt = opt
        # self.loss_weight1 = nn.Parameter(torch.tensor(1.0, dtype=float))
        # self.loss_weight2 = nn.Parameter(torch.tensor(1.0, dtype=float))
        self.classifiers = self.gcn_model.classifiers

    def weighted_loss(self, l1, l2, l3, l4, loss_weight=0.9):
        # return loss = loss + loss_weight * loss_i2r + loss_weight * loss_z2r  + loss_x2r
        return l1 + torch.sigmoid(self.loss_weight1)*l2 + torch.sigmoid(self.loss_weight2)*l3 + l4
        # return l1 + (torch.tanh(self.loss_weight1)+1)/2*l2 + (torch.tanh(self.loss_weight2)+1)/2*l3 + l4

    def conv_l2(self):
        return self.gcn_model.gcn.conv_l2()

    def forward(self, inputs, eval):
        main_logits, logits_dict, original_logits, inventions_logits, pooling_output = self.gcn_model(inputs, eval)
        return main_logits, logits_dict, original_logits, inventions_logits, pooling_output

class GCNRelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        embeddings = (self.emb, self.pos_emb, self.ner_emb)
        self.init_embeddings()

        # gcn layer
        if opt.get('aggcn', False):
            from model.aggcn import AGGCN
            self.gcn = AGGCN(opt, embeddings)
        else:
            self.gcn = GCN(opt, embeddings, opt['hidden_dim'], opt['num_layers'])
        self.in_drop = nn.Dropout(opt['input_dropout'])

        # output mlp layers
        in_dim = opt['hidden_dim']*3
        if self.opt.get('no_gcn', False):
            layers = [nn.Linear(in_dim*2, opt['hidden_dim']), nn.ReLU()]
        else:
            layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers']-1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]

        self.out_mlp = nn.Sequential(*layers)

        self.effect_type = self.opt['effect_type']

        print("Casual Effect is:{}".format(self.effect_type))

        self.criterion = nn.CrossEntropyLoss()
        self.ner_classifier = nn.Linear(opt['hidden_dim'], len(constant.NER_TO_ID))

        if self.effect_type == 'None':
            self.rnn = nn.LSTM(opt['emb_dim'] + opt['ner_dim'] + opt['pos_dim'], opt['rnn_hidden'], opt['rnn_layers'], batch_first=True,
                           dropout=opt['rnn_dropout'], bidirectional=True)
        else:
            self.rnn = nn.LSTM(opt['emb_dim'], opt['rnn_hidden'], opt['rnn_layers'], batch_first=True,
                           dropout=opt['rnn_dropout'], bidirectional=True)

        self.in_dim = opt['rnn_hidden'] * 2
        self.rnn_drop = nn.Dropout(opt['rnn_dropout'])
        self.ctx_I = opt['ctx_I']
        self.opt['context'] = self.opt.get('context', False)
        if self.effect_type != 'None':

            self.senttrans = nn.Linear(opt['emb_dim'], 2 * opt['emb_dim'])
#            self.sent2rel = nn.Linear(2 * opt['emb_dim'], opt['num_class'])
            if self.ctx_I:
                self.sent2rel = nn.Linear(opt['hidden_dim'], opt['num_class'])
            else:
                self.sent2rel = nn.Linear(opt['emb_dim'], opt['num_class'])
            self.tags2rel = nn.Linear(4 * opt['ner_dim'], opt['num_class'])
            #self.poss2rel = nn.Linear(2 * opt['ner_dim'], opt['num_class'])
            self.ents2rel = nn.Linear(3 * opt['hidden_dim'], opt['num_class'])
            ## NOTE: changed
            # self.ents2rel = nn.Linear(2 * opt['hidden_dim'], opt['num_class'])
            self.gate2rel = nn.Linear(3 * opt['hidden_dim'], opt['num_class'])
            self.sigmoid = nn.Sigmoid()
            #if self.opt.get('rnn', False):
            #input_size = self.in_dim
            self.register_buffer("untreated_spt", torch.zeros(3 * opt['hidden_dim']))
            self.average_ratio = 0.0005

            # counterfactual generation
            if self.opt.get('cfgen', False):
                layers = [nn.Linear(3*opt['hidden_dim'], 3*opt['hidden_dim']), nn.ReLU()]
                for _ in range(2):
                    layers += [nn.Linear(3*opt['hidden_dim'], 3*opt['hidden_dim']), nn.ReLU()]
                self.cf_mlp = nn.Sequential(*layers)

        self.main_classifier = nn.Linear(opt['hidden_dim'], opt['num_class'])

        ### baseline
        self.classifiers = [self.main_classifier, self.ner_classifier]
        if self.effect_type != "None":
            self.classifiers += [self.tags2rel, self.ents2rel, self.sent2rel, 
                            self.gate2rel]
        if self.opt.get('lws', False):
            self.tau = nn.Parameter(torch.tensor(self.opt['tau'], dtype=float))
        else:
            self.tau = self.opt.get('tau', 0)

    def moving_average(self, holder, input):
            if self.opt['mr']:
                assert len(input.shape) == 3
                with torch.no_grad():
                    holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).mean(0)
            else:
                assert len(input.shape) == 2

                with torch.no_grad():
                    holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
            return holder

    ### baseline
    def adaptive_classify(self, classifier, x):
        from utils.bert_utils import tau_classify
        if not self.opt.get('use_tau', False):
            return classifier(x)
        return tau_classify(x, classifier, self.tau)

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")

    def calculatelogits(self, I, X, Z, mask, pool_type):
        # I: embs (with context)
        # X: cat[subj_out, obj_out]
        # Z: ner_embs
        # Y: output
        # pooling
        # I-->Y
        #union_rep = torch.cat([I,X], dim=1)
        #union_rep = self.out_mlp(union_rep)
        #main_logits = self.main_classifier(union_rep)

        logitsdict = {}
        I = pool(I, mask, type=pool_type)
        # I = self.senttrans(I)
        # i2rlogits = self.sent2rel(I)
        i2rlogits = self.adaptive_classify(self.sent2rel, I)
        if self.opt['mr']:
            max_rel_num = X.shape[1]
            i2rlogits = i2rlogits.unsqueeze(1).expand(-1, max_rel_num, -1)
        logitsdict['i2rlogits'] = i2rlogits

        # Z1-->Y Z_NER
        # z2rlogits = self.tags2rel(Z)
        z2rlogits = self.adaptive_classify(self.tags2rel, Z)

        logitsdict['z2rlogits'] = z2rlogits

        # # Z2-->Y  Z_POS
        # zp2rlogits = self.poss2rel(Z2)
        # logitsdict['zp2rlogits'] = zp2rlogits

        # X-->Y
        # x2rlogits = self.ents2rel(X)
        x2rlogits = self.adaptive_classify(self.ents2rel, X)
        logitsdict['x2rlogits'] = x2rlogits

#        union_logits = torch.cat([i2rlogits, z2rlogits, x2rlogits], dim=1)
#        logits = self.union2rel(union_logits)

        if self.opt['fuse_type'] == 'SUM':
            logits = i2rlogits + z2rlogits + x2rlogits
        elif self.opt['fuse_type'] == 'GATED':
            gated_logits = self.gate2rel(X)
            logits = x2rlogits * torch.sigmoid(i2rlogits + z2rlogits + gated_logits)

        return  logitsdict, logits

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        new_mask =  masks.data.eq(constant.PAD_ID).long().sum(1).squeeze()
        if len(new_mask.shape) == 0:
            seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1))
        else:
            seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, inputs, eval):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        pool_type = self.opt['pooling']
        maxlen = max(l)
        batch_size = words.size(0)

        def inputs_to_tree_reps(head, words, l, prune, subj_pos, obj_pos):
            head, words, subj_pos, obj_pos = head.cpu().numpy(), words.cpu().numpy(), subj_pos.cpu().numpy(), obj_pos.cpu().numpy()
            trees = [head_to_tree(head[i], words[i], l[i], prune, subj_pos[i], obj_pos[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=False).reshape(1, maxlen, maxlen) for tree in trees]
            cf_paths = None
            if self.opt['context']:
                cf_paths = [get_shortest_path_tmp(adj[i].reshape(maxlen, maxlen), subj_pos[i], obj_pos[i], l[i]) for i in range(len(l))]
                cf_paths = [torch.LongTensor(i) for i in cf_paths]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            adj = Variable(adj.cuda()) if self.opt['cuda'] else Variable(adj)
            return adj, cf_paths

        # deprecated mr
        if self.opt['mr'] and self.opt['prune_k'] > 0:
            adj = []
            cf_paths = [[] for i in range(batch_size)]
            for i in range(subj_pos.size(1)):
                subj_pos_i = subj_pos[:, i]
                obj_pos_i = obj_pos[:, i]
                adj_i, cf_paths_i = inputs_to_tree_reps(head.data, words.data, l, self.opt['prune_k'], subj_pos_i.data, obj_pos_i.data)
                adj.append(adj_i.unsqueeze(1))
                for j in range(batch_size):
                    cf_paths[j].append(cf_paths_i[j])
            adj = torch.cat(adj, dim=1)
        else:
            adj, cf_paths = inputs_to_tree_reps(head.data, words.data, l, self.opt['prune_k'], subj_pos.data, obj_pos.data)

        if self.opt['context']:
            #NOTE: not implemented for mr
            cf_spath_mask = torch.zeros_like(words).bool()
            for i in range(len(cf_paths)):
                cf_spath_mask[i, :].index_fill_(0, cf_paths[i].cuda(), 1)

            cf_neighbor_mask = torch.zeros_like(words).bool()
            # neighbors = [get_neighbor(head.data[i], l[i], subj_pos[i], obj_pos[i]) for i in range(len(l))]
            # for i in range(len(neighbors)):
            #     cf_neighbor_mask[i, :].index_fill_(0, torch.cuda.LongTensor(neighbors[i]), 1)

            cf_context_mask = cf_spath_mask | cf_neighbor_mask
            # cf_context_mask = cf_spath_mask
            # cf_context_mask = cf_neighbor_mask
            cf_context_mask_inv = ~ cf_context_mask
        elif self.opt['dependency']:
            # Dependency mask for subj and obj

            subj_pos_indicator = (subj_pos == 0)
            obj_pos_indicator = (obj_pos == 0)
            cf_indicator = subj_pos_indicator | obj_pos_indicator
            cf_indicator = cf_indicator.long().unsqueeze(-1).repeat(1, 1, adj.size(-1)).bool()
            cf_adj_mask = ~ (cf_indicator | cf_indicator.transpose(1, 2))
            # print(cf_adj_mask[0])
            cf_adj_mask = cf_adj_mask.long()
            cf_adj = adj * cf_adj_mask
            

        word_embs = self.emb(words)

        if self.effect_type == 'None':
            if self.opt['ner_dim'] > 0:
                word_embs = torch.cat([word_embs,self.ner_emb(ner), self.pos_emb(pos)], dim=-1)
            word_embs = self.in_drop(word_embs)
        
        ctx_inputs = self.rnn_drop(self.encode_with_rnn(word_embs, masks, words.size()[0]))

        # if self.opt['mr'] and self.opt['prune_k'] > 0: # not completed
        #     h = []
        #     pool_mask = []
        #     h_out = []
        #     for i in range(adj.size(1)):
        #         h_i, pool_mask_i = self.gcn(adj[:, i], ctx_inputs)
        #         h_out_i = pool(h_i, pool_mask_i, type=pool_type)
        #         h.append(h_i.unsqueeze(1))
        #         h_out_i.append(h_out_i.unsqueeze(1))
        #         pool_mask.append(pool_mask_i.unsqueeze(1))
        #     h = torch.cat(h, dim=1)
        #     pool_mask = torch.cat(pool_mask, dim=1)
        #     h_out = torch.cat(h_out, dim=1)
        if self.opt.get('no_gcn', False):
            h = ctx_inputs
            new_mask =  masks.unsqueeze(2)
            h_out = pool(h, new_mask, type=pool_type)
        else:
            if self.opt.get('aggcn', False):
                h, pool_mask = self.gcn(adj, inputs)
            else:
                h, pool_mask = self.gcn(adj, ctx_inputs)
            h_out = pool(h, pool_mask, type=pool_type)

        if self.opt['mr']:
            subj_mask_m, obj_mask_m = subj_pos.eq(0).eq(0).unsqueeze(-1), obj_pos.eq(0).eq(0).unsqueeze(-1)  # invert mask
            max_num_rel = subj_mask_m.shape[1]
            subj_out = torch.stack([pool(h, subj_mask_m[:,r_no], type="max") for r_no in range(max_num_rel)], dim=1) #if we use the average pooling, the performance will be dropped dramatically.
            obj_out = torch.stack([pool(h, obj_mask_m[:,r_no], type="max") for r_no in range(max_num_rel)], dim = 1)
            h_out = h_out.unsqueeze(1).expand(-1,max_num_rel,self.opt['hidden_dim'])
        else:
            subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)  # invert mask
            subj_out = pool(h, subj_mask, type=pool_type)
            obj_out = pool(h, obj_mask, type=pool_type)

        logitsdict = None
        original_logits  = intervention_logits = None

        if self.opt['ner_loss'] == True: # there are mistakes here
            subj_ner_label = pool(ner, subj_mask.squeeze(-1), type=pool_type)
            obj_ner_label = pool(ner, obj_mask.squeeze(-1), type=pool_type)
            sub_ner_logits = self.ner_classifier(subj_out)
            obj_ner_logits = self.ner_classifier(obj_out)
            sub_ner_tag = torch.argmax(sub_ner_logits, dim=1)
            obj_ner_tag = torch.argmax(obj_ner_logits, dim=1)
            sub_ner_embs = self.ner_emb(sub_ner_tag)
            obj_ner_embs = self.ner_emb(obj_ner_tag)
            ner_embs = torch.cat([sub_ner_embs, obj_ner_embs], dim=1)
            sub_ner_loss = self.criterion(sub_ner_logits, subj_ner_label)
            obj_ner_loss = self.criterion(obj_ner_logits, obj_ner_label)
            ner_loss = sub_ner_loss + obj_ner_loss
        else:
            ner_loss = None

        if self.effect_type == 'None':
            outputs = torch.cat([h_out, subj_out, obj_out], dim=-1)
            outputs = self.out_mlp(outputs)
            # main_logits = self.main_classifier(outputs)
            main_logits = self.adaptive_classify(self.main_classifier, outputs)
        else:
            rel_rep = torch.cat([h_out, subj_out, obj_out], dim=-1)
            # rel_rep = torch.cat([subj_out, obj_out], dim=-1)

            ner_embs = self.ner_emb(ner)
            pos_embs = self.pos_emb(pos)

            if self.opt['mr']:
                max_num_rel = subj_mask_m.shape[1]
                sub_ner = torch.stack([pool(ner_embs, subj_mask_m[:, r_no], type="max") for r_no in range(max_num_rel)],dim=1)
                obj_ner = torch.stack([pool(ner_embs, obj_mask_m[:, r_no], type="max") for r_no in range(max_num_rel)],dim=1)
                sub_pos_rep = torch.stack([pool(pos_embs, subj_mask_m[:, r_no], type="max") for r_no in range(max_num_rel)],dim=1)
                obj_pos_rep = torch.stack([pool(pos_embs, obj_mask_m[:, r_no], type="max") for r_no in range(max_num_rel)],dim=1)

            else:
                sub_ner = pool(ner_embs, subj_mask, type=pool_type)
                obj_ner = pool(ner_embs, obj_mask, type=pool_type)

                sub_pos_rep = pool(pos_embs, subj_mask, type=pool_type)
                obj_pos_rep = pool(pos_embs, obj_mask, type=pool_type)

            ner_rep = torch.cat([sub_ner, obj_ner], dim=-1)
            pos_rep = torch.cat([sub_pos_rep, obj_pos_rep], dim=-1)

            tag_rep = torch.cat([ner_rep, pos_rep], dim=-1)

            if self.ctx_I:
                I_rep = h

            logitsdict, main_logits = self.calculatelogits(I_rep,rel_rep,tag_rep, pool_mask,pool_type)

            original_logits = main_logits
            intervention_logits = None

            if self.opt['cfgen']:
                average_x = self.cf_mlp(rel_rep)
                inter_logitsdict, intervention_logits = self.calculatelogits(I_rep,average_x, tag_rep,pool_mask, pool_type)
                logitsdict['cflogits'] = inter_logitsdict['x2rlogits']
                # main_logits = original_logits - intervention_logits

            if self.opt['mi']:  # mutltiple interventions.
                self.untreated_spt = self.moving_average(self.untreated_spt, rel_rep)

            if eval:
               if self.effect_type == 'TDE':
                    #rel_rep = torch.cat([h_out, subj_out, obj_out], dim=1)
                    #ner_rep = torch.cat([sub_ner, obj_ner], dim=1)
                    bz = rel_rep.shape[0]
                    dim = rel_rep.shape[-1]

                    if self.opt['mr']:
                        max_num_rel = subj_mask_m.shape[1]
                        average_x = torch.zeros(bz, max_num_rel, dim).cuda()
                    else:
                        if self.opt['cfgen']:
                            average_x = self.cf_mlp(rel_rep)
                        else:
                            average_x = torch.zeros(bz, dim).cuda()
                            
                    if self.opt['context']:
                        ### high level intervention
                        mask_inter = cf_context_mask_inv.unsqueeze(-1)
                        # mask_inter = cf_context_mask.unsqueeze(-1)
                        I_rep_inter = I_rep*mask_inter

                        ### low level intervention 
                        if self.opt['mr']:
                            # subj_mask_m, obj_mask_m = subj_pos.eq(0).eq(0).unsqueeze(-1), obj_pos.eq(0).eq(0).unsqueeze(-1)  # invert mask
                            word_embs_inter = self.emb(words)
                            # mask_inter = cf_context_mask_inv.unsqueeze(-1).bool() & subj_mask_m.bool() & obj_mask_m.bool() # use "and" for inv mask
                            # mask_inter = mask_inter.long()
                            # word_embs_inter = word_embs_inter*mask_inter
                            # ctx_inputs_inter = self.rnn_drop(self.encode_with_rnn(word_embs_inter, masks, words.size()[0]))
                            # h_inter, pool_mask = self.gcn(adj, ctx_inputs_inter)
                            # h_out_inter = pool(h_inter, pool_mask, type=pool_type)
                            # subj_out_inter = torch.stack([pool(h, subj_mask_m[:,r_no], type="max") for r_no in range(max_num_rel)], dim=1) #if we use the average pooling, the performance will be dropped dramatically.
                            # obj_out_inter = torch.stack([pool(h, obj_mask_m[:,r_no], type="max") for r_no in range(max_num_rel)], dim = 1)
                            # h_out = h_out.unsqueeze(1).expand(-1,max_num_rel,self.opt['hidden_dim'])
                            # average_x = torch.cat([h_out_inter, subj_out_inter, obj_out_inter], dim=1) # intervened rel_rep
                        else:
                            word_embs_inter = self.emb(words)

                            # Shortest Path design
                            mask_inter = cf_context_mask_inv.unsqueeze(-1).bool() & subj_mask.bool() & obj_mask.bool() # usage "and" for inv mask

                            mask_inter = mask_inter.long()
                            word_embs_inter = word_embs_inter*mask_inter  # masking at input word level
                            ctx_inputs_inter = self.rnn_drop(self.encode_with_rnn(word_embs_inter, masks, words.size()[0]))
                            h_inter, pool_mask = self.gcn(adj, ctx_inputs_inter)
                            h_out_inter = pool(h_inter, pool_mask, type=pool_type)
                            subj_out_inter = pool(h_inter, subj_mask, type=pool_type)
                            obj_out_inter = pool(h_inter, obj_mask, type=pool_type)
                            I_inter = h_inter

                            # Pick one according to the use of h_out_inter
                            average_x = torch.cat([h_out_inter, subj_out_inter, obj_out_inter], dim=-1) # intervened rel_rep
                            # average_x = torch.cat([subj_out_inter, obj_out_inter], dim=-1) # intervened rel_rep
                    elif self.opt['dependency']:
                            # apply counterfactual adj
                            h_inter, pool_mask = self.gcn(cf_adj, ctx_inputs)
                            h_out_inter = pool(h_inter, pool_mask, type=pool_type)
                            subj_out_inter = pool(h_inter, subj_mask, type=pool_type)
                            obj_out_inter = pool(h_inter, obj_mask, type=pool_type)
                            I_inter = h_inter
                            average_x = torch.cat([h_out_inter, subj_out_inter, obj_out_inter], dim=-1) # intervened rel_rep

                    tag_placeholder = torch.zeros_like(ner_rep)
                    average_tag = torch.zeros_like(tag_rep)
                    average_ner = torch.cat([tag_placeholder, pos_rep], dim=-1)
                    average_pos = torch.cat([ner_rep, tag_placeholder], dim=-1)
                    average_I = torch.zeros_like(I_rep)

                    # logitsdict, original_logits = self.calculatelogits(I_rep,rel_rep,tag_rep,pool_mask,pool_type)
                    
                    # PICK one for case study
                    # TDE intervention
                    inter_logitsdict, intervention_logits = self.calculatelogits(I_rep,average_x, tag_rep,pool_mask,pool_type) # baseline X
                    # inter_logitsdict, intervention_logits = self.calculatelogits(I_inter,average_x, tag_rep,pool_mask,pool_type)
                    # inter_logitsdict, intervention_logits = self.calculatelogits(average_I,rel_rep, tag_rep,pool_mask,pool_type) # case study 1 S
                    # inter_logitsdict, intervention_logits = self.calculatelogits(I_rep,rel_rep, average_tag,pool_mask,pool_type) # case study 2 NER + POS
                    # inter_logitsdict, intervention_logits = self.calculatelogits(I_rep,rel_rep, average_ner,pool_mask,pool_type) # case study 3 NER
                    # inter_logitsdict, intervention_logits = self.calculatelogits(I_rep,rel_rep, average_pos,pool_mask,pool_type) # case study 4 POS
                    # inter_logitsdict, intervention_logits = self.calculatelogits(average_I,average_x, tag_rep,pool_mask,pool_type) # case study 5

                    if self.opt['instance_class'] is not None:
                        instance_class = self.opt['instance_class']
                        instance_logits = []
                        # types = ['Main', 'TDE', 'AMP', 'X', 'S', 'NER+POS', 'NER','POS']
                        types = ['Main', 'Mask X', 'Mask S', 'Mask NER','Mask POS']
                        main_predict = main_logits.max(dim=1)[1]
                        main_right_prediction = main_predict == instance_class
                        # main_filter = main_right_prediction.nonzero().repeat(1, main_logits.size(1))
                        instance_logits.append(main_logits)
                        # instance_logits.append(original_logits - intervention_logits) 
                        # instance_logits.append(original_logits + logitsdict['x2rlogits'])
                        instance_logits.append(intervention_logits) # baseline X intervention
                        instance_logits.append(self.calculatelogits(average_I,rel_rep, tag_rep,pool_mask,pool_type)[1]) # case study 1 S
                        # instance_logits.append(self.calculatelogits(I_rep,rel_rep, average_tag,pool_mask,pool_type)[1]) # case study 2 NER + POS
                        instance_logits.append(self.calculatelogits(I_rep,rel_rep, average_ner,pool_mask,pool_type)[1]) # case study 3 NER
                        instance_logits.append(self.calculatelogits(I_rep,rel_rep, average_pos,pool_mask,pool_type)[1]) # case study 4 POS
                        if self.opt['dependency']:
                            types[1] = 'Mask Dependency'
                        elif self.opt['context']:
                            types[1] = 'Mask Context'
                            print(cf_context_mask)
                        intervention_predict = intervention_logits.max(dim=1)[1]
                        x_wrong_prediction = intervention_predict != instance_class
                        main_filter = main_right_prediction & x_wrong_prediction
                        main_filter = main_filter.nonzero()
                        print("sample indices:", main_filter.squeeze().data)
                        main_filter = main_filter.repeat(1, main_logits.size(1))

                        main_results = []
                        other_results = []
                        for i in range(len(instance_logits)):
                            # print(instance_logits[i].size(), main_filter.size())
                            instance = torch.gather(instance_logits[i], 0, main_filter)
                            temp = 2
                            instance = F.softmax(instance/temp, dim=1)
                            # print(instance[:, instance_class].data)
                            main_results.append(instance[:, instance_class].data)
                            other_classes = instance.clone()
                            other_classes[:, instance_class] = -1e20
                            # print(other_classes.max(dim=1)[0].data)
                            other_results.append(other_classes.max(dim=1)[0].data)
                        main_results = torch.stack(main_results, dim=0)
                        other_results = torch.stack(other_results, dim=0)
                        print(types)
                        print(main_results.T)
                        print(other_results.T)
                    
                    # Amplifier intervention
                    amp_inter_logitsdict, amp_intervention_logits = self.calculatelogits(average_I,rel_rep, average_tag,pool_mask,pool_type) # best
                    # inter_logitsdict, intervention_logits = self.calculatelogits(I_rep,average_x, average_tag,pool_mask,pool_type)
                    # inter_logitsdict, intervention_logits = self.calculatelogits(average_I,average_x, average_tag,pool_mask,pool_type)
  
                    # intervention_logits = logitsdict['i2rlogits'] + logitsdict['z2rlogits']
                    # average_emb = torch.zeros_like(I_rep).cuda()
                    # logitsdict, intervention_logits = self.calculatelogits(average_emb,average_x, tag_rep,pool_mask,pool_type)

                    # Main effect
                    # main_logits = original_logits     # vanilla baseline
                    # main_logits = original_logits - intervention_logits     # tde
                    main_logits = original_logits + amp_intervention_logits     # best 
                    # main_logits = original_logits - intervention_logits + amp_intervention_logits

                    if self.opt['mi']: # multiple interventions.
                        if self.opt['mr']:
                            max_num_rel = subj_mask_m.shape[1]
                            average_x_mi = self.untreated_spt.view(1, 1,dim).expand(bz, 1, dim).expand(bz, max_num_rel, dim)
                        else:
                            average_x_mi = self.untreated_spt.view(1, dim).expand(bz,dim)

                        logitsdict, inter_logits_mi = self.calculatelogits(I_rep, average_x_mi, tag_rep, pool_mask, pool_type)

                        main_logits_mi = original_logits - inter_logits_mi
                        main_logits += main_logits_mi

        return main_logits, logitsdict, original_logits, intervention_logits, h_out

class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """
    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()

        self.opt = opt
        self.effect_type = self.opt['effect_type']
        self.layers = num_layers
        self.use_cuda = opt['cuda']
        self.mem_dim = mem_dim
        self.opt['use_bert'] = self.opt.get('use_bert', False)
        if self.opt['use_bert']:
            self.in_dim = opt['bert_dim'] + opt['pos_dim'] + opt['ner_dim']
        else:
            self.in_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim']

        self.emb, self.pos_emb, self.ner_emb = embeddings

        # rnn layer
        if self.opt.get('rnn', False):
            input_size = self.in_dim
            self.rnn = nn.LSTM(input_size, opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
                    dropout=opt['rnn_dropout'], bidirectional=True)
            self.in_dim = opt['rnn_hidden'] * 2
            self.rnn_drop = nn.Dropout(opt['rnn_dropout']) # use on last layer output

        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        new_mask =  masks.data.eq(constant.PAD_ID).long().sum(1).squeeze()
        if len(new_mask.shape) == 0:
            seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1))
        else:
            seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs):
#        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack
#        if self.effect_type != 'None':
#         word_embs = self.emb(words)
#         embs = [word_embs]

       # if self.effect_type == 'None':
       #  if self.opt['pos_dim'] > 0:
       #      embs += [self.pos_emb(pos)]
       #  if self.opt['ner_dim'] > 0:
       #      embs += [self.ner_emb(ner)]
       #
       #  ner_embs = self.ner_emb(ner)
       #  pos_embs = self.pos_emb(pos)
       #
       #  embs = torch.cat(embs, dim=2)
       #  embs = self.in_drop(embs)

        # rnn layer
        # if self.opt.get('rnn', False):
        #     gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, masks, words.size()[0]))
        # else:
        #     gcn_inputs = embs

        # gcn layer
        #embs, ner_embs, pos_embs = None

        gcn_inputs = inputs

        denom = adj.sum(2).unsqueeze(2) + 1
        mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        # zero out adj for ablation
        if self.opt.get('no_adj', False):
            adj = torch.zeros_like(adj)

        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs) # self loop
            AxW = AxW / denom

            gAxW = F.relu(AxW)
            gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW

        #return gcn_inputs, mask, embs, ner_embs, pos_embs
        return gcn_inputs, mask

def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0

