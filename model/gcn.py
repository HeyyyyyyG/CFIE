
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np


from utils import constant, torch_utils
from neuronlp2.nn import CharCNN, ChainCRF

class NERClassifier(nn.Module):
    def __init__(self, opt, emb_matrix=None, char_emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.char_emb_matrix = char_emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.emb_matrix = torch.from_numpy(self.emb_matrix)
        self.emb.weight.data.copy_(self.emb_matrix)

        self.deprel_emb = nn.Embedding(len(constant.DEPREL_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.char_emb = nn.Embedding(opt['char_vocab_size'], opt['char_emb_dim'], padding_idx=constant.PAD_ID)
        self.char_emb_matrix = torch.from_numpy(self.char_emb_matrix)
        self.char_emb.weight.data.copy_(self.char_emb_matrix)

        self.dropout_in = nn.Dropout(0.3)


        self.effect_type = self.opt['effect_type']

        print("Casual Effect is:{}".format(self.effect_type))

        self.char_cnn = CharCNN(2, opt['char_emb_dim'], opt['char_emb_dim'], hidden_channels=4 * opt['char_emb_dim'], activation="elu")


        self.bilstm = nn.LSTM(2*(opt['char_emb_dim'] + opt['emb_dim'])+opt['pos_dim'], opt['hidden_dim'], num_layers= self.opt['rnn_layers'],
                            dropout = 0.3,
                            batch_first=True,
                            bidirectional=True)


        self.middle_mlp = nn.Linear(2*opt['hidden_dim'],2*opt['hidden_dim'])
        # self.middle_mlp2 = nn.Linear(2*opt['hidden_dim'],2*opt['hidden_dim'])
        # self.middle_mlp3 = nn.Linear(2*opt['hidden_dim'],2*opt['hidden_dim'])
        self.gcn = GCN(opt, 2* opt['hidden_dim'], opt['num_gcn_layers'])
        self.ent2ner = nn.Linear(2* opt['hidden_dim'], opt['num_class'])
        self.sent2ner = nn.Linear(opt['emb_dim'], opt['num_class'])
        self.pos2ner = nn.Linear(opt['pos_dim'], opt['num_class'])
        self.main_classifier = nn.Linear(2*opt['hidden_dim'], opt['num_class'])



    def calculatelogits(self, I, X, Z):

        logitsdict = {}


        i2rlogits = self.sent2ner(I)
        logitsdict['i2rlogits'] = i2rlogits

        # Z-->Y

        z2rlogits = self.pos2ner(Z)
        logitsdict['z2rlogits'] = z2rlogits
        # X-->Y

        x2rlogits = self.ent2ner(X)
        logitsdict['x2rlogits'] = x2rlogits


        logits = i2rlogits +  x2rlogits + z2rlogits


        return  logitsdict, logits

    def forward(self, inputs, eval):
        tokens, mask, chars_seq, pos,deprel, head, subj_pos, obj_pos= inputs

        word = self.emb(tokens)
        char = self.char_emb(chars_seq)


        char = self.char_cnn(char)
        word = self.dropout_in(word)
        char = self.dropout_in(char)
        pos_embs = self.dropout_in(self.pos_emb(pos))

        origin_enc = torch.cat([word, char], dim=2)

        batch_size = origin_enc.shape[0]
        sent_len = origin_enc.shape[1]
        hsize = origin_enc.shape[2]

        dep_head_emb = torch.gather(origin_enc, 1, head.view(batch_size, sent_len, 1).expand(batch_size, sent_len, hsize))
        dep_emb = self.deprel_emb(deprel)
        enc = torch.cat((origin_enc, dep_head_emb, dep_emb), 2)


        if mask is not None:
            length = mask.sum(dim=1).long()

            packed_enc = pack_padded_sequence(enc, length, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.bilstm(packed_enc)
            output, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            output, _ = self.bilstm(enc)



        logitsdict = None
        h_out = word

        output = self.middle_mlp(output)

        if self.effect_type == 'None':
            logitsdict, main_logits = self.calculatelogits(h_out, output, pos_embs)


        else:
            logitsdict, main_logits = self.calculatelogits(h_out, output, pos_embs)

            if eval:
               if self.effect_type == 'CAUSAL':
                    average_x = self.get_counterfact(origin_enc, mask, deprel,head,'headall')

                    logitsdict, original_logits = self.calculatelogits(h_out,output,pos_embs)
                    logitsdict, TDE_logtis = self.calculatelogits(h_out,average_x, pos_embs)

                    main_logits =original_logits - TDE_logtis + self.opt['alpha']*logitsdict['x2rlogits']

        return main_logits, logitsdict

    def get_counterfact(self, enc, mask, deprel, head, entity_or_head):

        batch_size = enc.shape[0]
        sent_len = enc.shape[1]
        hsize = enc.shape[2]

        dep_head_emb = torch.gather(enc, 1, head.view(batch_size, sent_len, 1).expand(batch_size, sent_len, hsize))
        dep_emb = self.deprel_emb(deprel)


        if entity_or_head=='oneall':
            out = []
            for j in range(sent_len):
                new_enc = torch.cat([enc,dep_head_emb,dep_emb],dim=2)
                new_enc[:,j] = torch.zeros(new_enc.shape[0],new_enc.shape[2])

                if mask is not None:
                    # prepare packed_sequence
                    length = mask.sum(dim=1).long()

                    packed_enc = pack_padded_sequence(new_enc, length, batch_first=True, enforce_sorted=False).cuda()

                    packed_out, _ = self.bilstm(packed_enc)
                    output, _ = pad_packed_sequence(packed_out, batch_first=True)

                    output = output[:,j].view(output.shape[0],1,output.shape[2])
                    out.append(output)
            output = torch.cat([i for i in out], dim=1).cuda()
        elif entity_or_head=='headall':
            out = []
            for j in range(sent_len):
                new_enc = torch.cat([enc,dep_head_emb,dep_emb],dim=2)
                n=[]
                for i in range(batch_size):
                    masktensor = new_enc[i]

                    masktensor[head[i][j]]=torch.zeros(new_enc.shape[2])
                    n.append(masktensor.unsqueeze(dim=0))
                new_enc = torch.cat([i for i in n],dim=0)

                if mask is not None:
                    # prepare packed_sequence
                    length = mask.sum(dim=1).long()

                    packed_enc = pack_padded_sequence(new_enc, length, batch_first=True, enforce_sorted=False).cuda()

                    packed_out, _ = self.bilstm(packed_enc)
                    output, _ = pad_packed_sequence(packed_out, batch_first=True)

                    output = output[:,j].view(output.shape[0],1,output.shape[2])


                    out.append(output)
            output = torch.cat([i for i in out], dim=1).cuda()
        elif entity_or_head=='entityheadall':
            out = []
            for j in range(sent_len):
                new_enc = torch.cat([enc,dep_head_emb,dep_emb],dim=2)
                n=[]
                for i in range(batch_size):
                    masktensor = new_enc[i]
                    masktensor[j]=torch.zeros(new_enc.shape[2])
                    masktensor[head[i][j]]=torch.zeros(new_enc.shape[2])
                    n.append(masktensor.unsqueeze(dim=0))
                new_enc = torch.cat([i for i in n],dim=0)

                if mask is not None:
                    # prepare packed_sequence
                    length = mask.sum(dim=1).long()

                    packed_enc = pack_padded_sequence(new_enc, length, batch_first=True, enforce_sorted=False).cuda()

                    packed_out, _ = self.bilstm(packed_enc)
                    output, _ = pad_packed_sequence(packed_out, batch_first=True)

                    output = output[:,j].view(output.shape[0],1,output.shape[2])


                    out.append(output)
            output = torch.cat([i for i in out], dim=1).cuda()

        else:
             raise Exception('invalid mask')

        output = self.middle_mlp(output)
        return output

   

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0


class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """
    def __init__(self, opt,  mem_dim, num_layers):
        super(GCN, self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.use_cuda = opt['cuda']
        self.mem_dim = mem_dim
        self.in_dim = 2 * opt['hidden_dim']


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
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, gcn_inputs):


        # gcn layer
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

        return gcn_inputs, mask
