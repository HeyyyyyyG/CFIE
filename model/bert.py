
from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.gcn import GCN, pool
from model.tree import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils
from utils.bert_utils import tau_classify

class BERTRelationModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        # bert
        self.multi_piece = opt['multi_piece']
        self.bert_encoder = BertModel.from_pretrained(opt['bert_model'])
        self.bert_encoder.resize_token_embeddings(self.bert_encoder.config.vocab_size+4) 
        if 'freeze' in opt.keys() and opt['freeze']:
            for param in self.bert_encoder.parameters():
                param.requires_grad = False
        self.bert_dim = self.bert_encoder.config.hidden_size
        opt['bert_dim'] = self.bert_dim
        self.hidden_dim = self.bert_dim
        self.cls_dropout = nn.Dropout(0.1)  # dropout on CLS transformed token embedding
        self.ent_dropout = nn.Dropout(0.1)  # dropout on average entity embedding

        # classifier balance
        self.tau = self.opt.get('tau', 1)
        self.use_tau = self.opt.get('use_tau', False)

        # create embedding layers
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        self.init_embeddings()

        # NOTE: gcn layer, disabled
        # embeddings = (None, self.pos_emb, self.ner_emb)
        # self.gcn = GCN(opt, embeddings, opt['hidden_dim'], opt['num_layers'])
        # self.hidden_dim = opt['hidden_dim']

        self.effect_type = self.opt['effect_type']

        print("Casual Effect is:{}".format(self.effect_type))

        self.criterion = nn.CrossEntropyLoss()
        self.ner_classifier = nn.Linear(self.hidden_dim, len(constant.NER_TO_ID))
        self.main_classifier = nn.Linear(self.hidden_dim*3, opt['num_class'])
        self.classifiers = [self.main_classifier, self.ner_classifier]

        # layers = []
        # for _ in range(self.opt['mlp_layers']):
        #     layers += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()]
        # self.out_mlp = nn.Sequential(*layers)

        self.ctx_I = opt['ctx_I']
        if self.effect_type != 'None':
            # mlp to further encode sent and ent reps, usually not used
            layers = []
            for _ in range(self.opt['mlp_layers']):
                layers += [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()]
            self.out_mlp = nn.Sequential(*layers)

            # self.senttrans = nn.Linear(opt['emb_dim'], 2 * opt['emb_dim'])
            if self.ctx_I:
                self.sent2rel_classifier = nn.Linear(self.hidden_dim, opt['num_class'])
            else:
                self.sent2rel_classifier = nn.Linear(self.hidden_dim, opt['num_class'])
            self.tags2rel_classifier = nn.Linear(4 * opt['ner_dim'], opt['num_class'])
            #self.poss2rel = nn.Linear(2 * opt['ner_dim'], opt['num_class'])
            self.ents2rel_classifier = nn.Linear(3 * self.hidden_dim, opt['num_class'])
            # self.ents2rel_classifier = nn.Linear(2 * self.hidden_dim, opt['num_class'])
            if opt['fuse_type'] == 'GATED':
                self.gate2rel = nn.Linear(3 * self.hidden_dim, opt['num_class'])

            # trakc all classifiers
            self.classifiers.extend([self.sent2rel_classifier, self.tags2rel_classifier, self.ents2rel_classifier])

            # counterfactual generation
            if self.opt['cfgen']:
                layers = [nn.Linear(3*self.hidden_dim, 3*self.hidden_dim), nn.ReLU()]
                for _ in range(2):
                    layers += [nn.Linear(3*self.hidden_dim, 3*self.hidden_dim), nn.ReLU()]
                self.cf_mlp = nn.Sequential(*layers)

        self.slps = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim) for i in range(3)])


    def decoupled_freeze(self):
        # freeze all parameters except the classifiers
        for param_name, param in self.named_parameters():
            # Freeze all parameters except self attention parameters
            if 'classifier' not in param_name:
                param.requires_grad = False
            # print('  | ', param_name, param.requires_grad)

    def reset_classifiers(self):
        for i in self.classifiers:
            i.reset_parameters()

    def adaptive_classify(self, classifier, x):
        from utils.bert_utils import tau_classify
        if not self.use_tau:
            return classifier(x)
        return tau_classify(x, classifier, self.tau)

    def compute_rel_pre_features(self, H0, H1, H2):
        H0 = self.slps[0](torch.tanh(H0))
        H1 = self.slps[1](torch.tanh(H1))
        H2 = self.slps[2](torch.tanh(H2))
        return H0, H1, H2

    def init_embeddings(self):
        pass

    def calculatelogits(self, I, X, Z, pool_type):
        # I: embs (with context)
        # X: cat[subj_out, obj_out]
        # Z: ner_embs
        # Y: output
        # pooling
        # I-->Y

        logitsdict = {}
        # I = self.senttrans(I)
        i2rlogits = self.sent2rel_classifier(I)
        logitsdict['i2rlogits'] = i2rlogits

        # Z1-->Y Z_NER
        z2rlogits = self.tags2rel_classifier(Z)
        logitsdict['z2rlogits'] = z2rlogits

        # # Z2-->Y  Z_POS
        # zp2rlogits = self.poss2rel(Z2)
        # logitsdict['zp2rlogits'] = zp2rlogits

        # X-->Y
        x2rlogits = self.ents2rel_classifier(X)
        logitsdict['x2rlogits'] = x2rlogits

        if self.opt['fuse_type'] == 'SUM':
            logits = i2rlogits + z2rlogits + x2rlogits
        elif self.opt['fuse_type'] == 'GATED':
            gated_logits = self.gate2rel(X)
            logits = x2rlogits * torch.sigmoid(i2rlogits + z2rlogits + gated_logits)

        return  logitsdict, logits

    def calculatelogits_bert(self, I, X, Z, pool_type):
        logitsdict = {}
        main_rep = torch.cat([I, X], dim=-1)
        main_logits = self.adaptive_classify(self.main_classifier, main_rep)
        logitsdict['main_logits'] = main_logits

        # Z-->Y 
        z2rlogits = self.tags2rel_classifier(Z)
        logitsdict['z2rlogits'] = z2rlogits

        logits = main_logits + z2rlogits
        return  logitsdict, logits

    def extract_entity(self, sequence_output, e_mask):
        extended_e_mask = e_mask.unsqueeze(1)
        extended_e_mask = torch.bmm(
            extended_e_mask.float(), sequence_output).squeeze(1)
        return extended_e_mask.float()

    def forward(self, inputs, eval):
        bert_inputs, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type = inputs # unpack
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2) # invert mask
        pool_type = self.opt['pooling']

        # bert
        input_ids = bert_inputs['input_ids']
        batch_size = input_ids.size(0)
        attention_mask = bert_inputs['attention_mask']
        bert_outputs = self.bert_encoder(input_ids, attention_mask)[0]
        # NOTE: must extract this first before offsetting
        cls_rep = bert_outputs[:, 0]
        sent_rep = self.cls_dropout(cls_rep)

        ###### ONEIE ###
        # token_lens = bert_inputs['token_lens']
        # if self.multi_piece == 'first':
        #     # select the first piece for multi-piece words
        #     offsets = token_lens_to_offsets(token_lens)
        #     offsets = input_ids.new(offsets)
        #     # + 1 because the first vector is for [CLS]
        #     offsets = offsets.unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
        #     bert_outputs = torch.gather(bert_outputs, 1, offsets)
        # elif self.multi_piece == 'average':
        #     # average all pieces for multi-piece words
        #     idxs, masks, token_num, token_len = token_lens_to_idxs(token_lens)
        #     idxs = input_ids.new(idxs).unsqueeze(-1).expand(batch_size, -1, self.bert_dim) + 1
        #     masks = bert_outputs.new(masks).unsqueeze(-1)
        #     bert_outputs = torch.gather(bert_outputs, 1, idxs) * masks
        #     bert_outputs = bert_outputs.view(batch_size, token_num, token_len, self.bert_dim)
        #     bert_outputs = bert_outputs.sum(2)
        # else:
        #     raise ValueError('Unknown multi-piece token handling strategy: {}'
        #                      .format(self.multi_piece))
        #####
        # subj_out = pool(bert_outputs, subj_mask, type=pool_type)
        # obj_out = pool(bert_outputs, obj_mask, type=pool_type)
        # subj_out = self.ent_dropout(subj_out)
        # obj_out = self.ent_dropout(obj_out)
        #####

        ##### R-BERT
        e1_mask = bert_inputs['e1_mask']
        e2_mask = bert_inputs['e2_mask']
        subj_out = self.ent_dropout(self.extract_entity(bert_outputs, e1_mask))
        obj_out = self.ent_dropout(self.extract_entity(bert_outputs, e2_mask))
        #####

        # GCN component
        # def inputs_to_tree_reps(head, words, l, prune, subj_pos, obj_pos):
        #     head, words, subj_pos, obj_pos = head.cpu().numpy(), words.cpu().numpy(), subj_pos.cpu().numpy(), obj_pos.cpu().numpy()
        #     trees = [head_to_tree(head[i], words[i], l[i], prune, subj_pos[i], obj_pos[i]) for i in range(len(l))]
        #     adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=False).reshape(1, maxlen, maxlen) for tree in trees]
        #     adj = np.concatenate(adj, axis=0)
        #     adj = torch.from_numpy(adj)
        #     return Variable(adj.cuda()) if self.opt['cuda'] else Variable(adj)

        # adj = inputs_to_tree_reps(head.data, words.data, l, self.opt['prune_k'], subj_pos.data, obj_pos.data)

        # ctx_inputs = bert_inputs
        # if self.opt['ner_dim'] > 0:
        #     ctx_inputs = torch.cat([ctx_inputs,self.ner_emb(ner), self.pos_emb(pos)], dim=-1)

        # h, pool_mask = self.gcn(adj, ctx_inputs)
        # h_out = pool(h, pool_mask, type=pool_type)
        # if h.shape[1] != subj_mask.shape[1]:
        #     print("debug")

        logitsdict = None
        original_logits  = intervention_logtis = None

        sent_rep, subj_out, obj_out = self.compute_rel_pre_features(sent_rep, subj_out, obj_out)
        if self.effect_type == 'None':
            outputs = torch.cat([sent_rep, subj_out, obj_out], dim=1)
            main_logits = self.adaptive_classify(self.main_classifier, outputs)
        else:
            # rel_rep = torch.cat([sent_rep, subj_out, obj_out], dim=1)
            ent_rep = torch.cat([sent_rep, subj_out, obj_out], dim=1)
            # ent_rep = self.out_mlp(ent_rep)

            ner_embs = self.ner_emb(ner)
            pos_embs = self.pos_emb(pos)

            sub_ner = pool(ner_embs, subj_mask, type=pool_type)
            obj_ner = pool(ner_embs, obj_mask, type=pool_type)

            sub_pos_rep = pool(pos_embs, subj_mask, type=pool_type)
            obj_pos_rep = pool(pos_embs, obj_mask, type=pool_type)

            ner_rep = torch.cat([sub_ner, obj_ner], dim=1)
            pos_rep = torch.cat([sub_pos_rep, obj_pos_rep], dim=1)

            tag_rep = torch.cat([ner_rep, pos_rep], dim=-1)

            # if self.ctx_I:
            #     bert_outputs = h

            
            if eval:
                if self.effect_type == 'TDE':
#                    rel_rep = torch.cat([h_out, subj_out, obj_out], dim=1)
                    #ner_rep = torch.cat([sub_ner, obj_ner], dim=1)
                    bz = ent_rep.shape[0]
                    dim = ent_rep.shape[1]
                    if self.opt['cfgen']:
                        average_x = self.cf_mlp(ent_rep)
                    else:
                        average_x = torch.zeros(bz, dim).cuda()

                    average_tag = torch.zeros_like(tag_rep).cuda()
                    average_I = torch.zeros_like(cls_rep).cuda()

                    logitsdict, original_logits = self.calculatelogits(cls_rep,ent_rep,tag_rep,pool_type)
                    # PICK one
                    logitsdict, intervention_logtis = self.calculatelogits(cls_rep,average_x, tag_rep,pool_type)
                    # logitsdict, intervention_logtis = self.calculatelogits(average_I,ent_rep, average_tag,pool_type)
                    # NOTE: changed
                    
                    # intervention_logtis = logitsdict['i2rlogits'] + logitsdict['z2rlogits']
                    # PICK one
                    main_logits = original_logits - intervention_logtis
                    # main_logits = original_logits + intervention_logtis
                else:
                    logitsdict, main_logits = self.calculatelogits(cls_rep,ent_rep,tag_rep,pool_type)
            else:
                logitsdict, main_logits = self.calculatelogits(cls_rep,ent_rep,tag_rep,pool_type)
                original_logits = main_logits
                if self.opt['cfgen']:
                    average_x = self.cf_mlp(ent_rep)
                    inter_logitsdict, intervention_logtis = self.calculatelogits(cls_rep,average_x, tag_rep,pool_type)
                    logitsdict['cflogits'] = inter_logitsdict['x2rlogits']
                    main_logits = original_logits - intervention_logtis

        return main_logits, logitsdict, original_logits, intervention_logtis, cls_rep

# Acknowledgement: Credit to ONEIE project from UIUC. 

def token_lens_to_offsets(token_lens):
    """Map token lengths to first word piece indices, used by the sentence
    encoder.
    :param token_lens (list): token lengths (word piece numbers)
    :return (list): first word piece indices (offsets)
    """
    max_token_num = max([len(x) for x in token_lens])
    offsets = []
    for seq_token_lens in token_lens:
        seq_offsets = [0]
        for l in seq_token_lens[:-1]:
            seq_offsets.append(seq_offsets[-1] + l)
        offsets.append(seq_offsets + [-1] * (max_token_num - len(seq_offsets)))
    return offsets

def token_lens_to_idxs(token_lens):
    """Map token lengths to a word piece index matrix (for torch.gather) and a
    mask tensor.
    For example (only show a sequence instead of a batch):

    token lengths: [1,1,1,3,1]
    =>
    indices: [[0,0,0], [1,0,0], [2,0,0], [3,4,5], [6,0,0]]
    masks: [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.33, 0.33, 0.33], [1.0, 0.0, 0.0]]

    Next, we use torch.gather() to select vectors of word pieces for each token,
    and average them as follows (incomplete code):

    outputs = torch.gather(bert_outputs, 1, indices) * masks
    outputs = bert_outputs.view(batch_size, seq_len, -1, self.bert_dim)
    outputs = bert_outputs.sum(2)

    :param token_lens (list): token lengths.
    :return: a index matrix and a mask tensor.
    """
    max_token_num = max([len(x) for x in token_lens])
    max_token_len = max([max(x) for x in token_lens])
    idxs, masks = [], []
    for seq_token_lens in token_lens:
        seq_idxs, seq_masks = [], []
        offset = 0
        for token_len in seq_token_lens:
            seq_idxs.extend([i + offset for i in range(token_len)]
                            + [-1] * (max_token_len - token_len))
            seq_masks.extend([1.0 / token_len] * token_len
                             + [0.0] * (max_token_len - token_len))
            offset += token_len
        seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens)))
        seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
        masks.append(seq_masks)
    return idxs, masks, max_token_num, max_token_len
