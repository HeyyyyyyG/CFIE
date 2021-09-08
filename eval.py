"""
Run evaluation with saved models.
"""
import random
import argparse
from tqdm import tqdm
import torch
import numpy as np
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
parser.add_argument('--model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/atis')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')

parser.add_argument('--effect_type', type=str, default='None', help="None/CAUSAL")
parser.add_argument('--alpha', type=float, default=1.2, help='alpha')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)

opt['alpha'] = args.alpha

opt['effect_type']=args.effect_type
print(opt)
emb_file = opt['vocab_dir'] + '/embedding.npy'
emb_matrix = np.load(emb_file)

char_emb_file = opt['vocab_dir'] + '/charembedding.npy'
char_emb_matrix = np.load(char_emb_file)

trainer = NERTrainer(opt,emb_matrix,char_emb_matrix)
trainer.load(model_file)

# load vocab
vocab_file = args.model_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
char_vocab_file = args.model_dir + '/charvocab.pkl'
charvocab = Vocab(char_vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."
assert opt['char_vocab_size'] == charvocab.size, "Char vocab size must match that in the saved model."


# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))

batch = NERDataLoader(data_file, opt['batch_size'], opt, vocab,charvocab, evaluation=True)

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

all_predictions = []
all_probs = []
batch_iter = tqdm(batch)
f_p =0
f_total_predict = 0
f_total_entity = 0
key=set()
prediction=set()
for i, b in enumerate(batch_iter):
    preds = trainer.predict(b,eval=True)
    word_seq_len = b[1].long().sum(1).squeeze()
    for idx in range(len(preds)):
        all_predictions+=preds[idx][:word_seq_len[idx]].tolist()
    array, output_spans, predict_spans = evaluate_num(preds, b[9], b[1].long().sum(1).squeeze(), id2label)
    p=array[0]
    total_predict=array[1]
    total_entity=array[2]
    key.update(output_spans)
    prediction.update(predict_spans)


all_predictions = [id2label[p] for p in all_predictions]
all_key = [id2label[p] for p in batch.gold()]

scorer.score_span(key, prediction,verbose=True)


