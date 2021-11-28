"""
Run evaluation with saved models.
"""
import random
import argparse
from tqdm import tqdm
import torch

from data.loader import DataLoader, BertDataLoader
from model.trainer import GCNTrainer, GCNBertTrainer
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--use_bert', default=False, action='store_true')
parser.add_argument('--use_tau', default=False, action='store_true')
parser.add_argument('--tau', type=float, default=1)
parser.add_argument('--context', default=False, action='store_true')
parser.add_argument('--dependency', action='store_true')
parser.add_argument('--lws', default=False, action='store_true')
parser.add_argument('--instance_class', type=int, default=None)
# parser.add_argument('--crt', action='store_true', default=False)
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
## baseline
opt['use_tau'] = args.use_tau
opt['tau'] = args.tau
opt['cuda'] = args.cuda
if 'lws' in model_file:
    opt['lws'] = True
    opt['use_tau'] = True
opt['context'] = args.context
opt['dependency'] = args.dependency
opt['instance_class'] = args.instance_class
if args.use_bert:
    trainer = GCNBertTrainer(opt)
else:
    trainer = GCNTrainer(opt)
trainer.load(model_file)

print('# parameters', sum(p.numel() for p in trainer.parameters))

# load vocab
vocab_file = args.model_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
if args.use_bert:
    DataLoader = BertDataLoader
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True, mr=opt['mr'])
train_file = opt['data_dir'] + '/{}.json'.format('train')
train_stat = DataLoader(train_file, opt['batch_size'], opt, vocab, evaluation=True, mr=opt['mr'])

helper.print_config(opt)
label2id = constant.LABEL_TO_ID
id2label = dict([(v,k) for k,v in label2id.items()])

predictions = []
all_probs = []
batch_iter = tqdm(batch)
effect_scores = 0
for i, b in enumerate(batch_iter):
    if opt['mr']:
        preds, probs, _, _, _, _, _, rel_nums = trainer.predict(b, eval=True)
        batch_preds = []
        batch_probs = []
        for p, rel in zip(preds, rel_nums):
            batch_preds += p[:rel]

        for p, rel in zip(probs, rel_nums):
            batch_probs += p[:rel]
            
        assert len(batch_preds) == sum(rel_nums)
        assert len(batch_probs) == sum(rel_nums)
        predictions += batch_preds
        all_probs += batch_probs
    else:
        preds, probs, _, _, _, _, _, batch_effect_score = trainer.predict(b, eval=True)
        predictions += preds
        all_probs += probs
        effect_scores += batch_effect_score

effect_scores /= len(batch_iter)
print("Final effect score: \n {}\n".format(effect_scores))
exl_labels = []
# exl_labels = [4, 9, 19, 20, 21]
all_labels = list(range(24))
if len(exl_labels) > 0:
    in_labels = [id2label[i] for i in all_labels if i not in exl_labels]
else:
    in_labels = None

predictions = [id2label[p] for p in predictions]
p, r, f1 = scorer.score(batch.gold(), predictions, all_probs, opt['dataset'], verbose=True, train_stat=train_stat.gold(), relation_filter=in_labels)
#p, r, f1 = scorer.score(batch.gold(), predictions, 'nyt', verbose=True)

from sklearn.metrics import f1_score, recall_score, precision_score
mean_f1 = f1_score(batch.gold(), predictions, average='macro', labels=in_labels)
mean_r = recall_score(batch.gold(), predictions, average='macro', labels=in_labels)
mean_p = precision_score(batch.gold(), predictions, average='macro', labels=in_labels)
print( "Precision (macro): {:.3%}".format(mean_p) )
print( "   Recall (macro): {:.3%}".format(mean_r) )
print( "       F1 (macro): {:.3%}".format(mean_f1) )

print("{} set evaluate result: {:.2f}\t{:.2f}\t{:.2f}".format(args.dataset,p,r,f1))

print("Evaluation ended.")

