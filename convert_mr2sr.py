import json
import sys, os
from copy import deepcopy

sr_subj_keys = ['subj_start', 'subj_end', 'subj_type']
sr_obj_keys = ['obj_start', 'obj_end', 'obj_type']

task = 'nyt24'
task = 'webnlg'
in_dir = 'dataset/'+task
out_dir = 'dataset/'+task+'_sr'
data_splits = ['train', 'dev', 'test']
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
for split in data_splits:
    ifilename = in_dir + '/' + split + '.json'
    ofilename = out_dir + '/' + split + '.json'
    with open(ifilename) as infile, open(ofilename, 'w') as outfile:
        data = json.load(infile)
        sr_data = []
        for datum in data:
            rels = datum['relation']
            for i in range(len(rels)):
                new_datum = deepcopy(datum)
                for subj_key in sr_subj_keys:
                    if 'type' in subj_key:
                        new_datum[subj_key] = datum[subj_key][i]
                    else:
                        new_datum[subj_key] = int(datum[subj_key][i])
                for obj_key in sr_obj_keys:
                    if 'type' in obj_key:
                        new_datum[obj_key] = datum[obj_key][i]
                    else:
                        new_datum[obj_key] = int(datum[obj_key][i])
                new_datum['relation'] = rels[i]
                sr_data.append(new_datum)
        json.dump(sr_data, outfile)
                
                
