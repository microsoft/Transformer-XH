# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# the specific stuff of fever

import torch
from torch.utils.data import Dataset, DataLoader
import dgl
from dgl import DGLGraph
import dgl.function as fn
from .base import TransformerXHDataset
from .utils import truncate_input_sequence



def batcher_fever(device):
    def batcher_dev(batch):

        graph, qid, label = batch[0]
        batch_graphs = dgl.batch([batch[0][0]])
        batch_graphs.ndata['encoding'] = batch_graphs.ndata['encoding'].to(device)
        batch_graphs.ndata['encoding_mask'] = batch_graphs.ndata['encoding_mask'].to(device)
        batch_graphs.ndata['segment_id'] = batch_graphs.ndata['segment_id'].to(device)
        qid = [batch[0][1]]
        label = [batch[0][2]]


        return batch_graphs, batch_graphs.ndata['label'].to(device), torch.tensor(label, dtype=torch.long).to(device), qid
    return batcher_dev





def encode_sequence_fever(question, title, passage, max_seq_len, tokenizer):
    
    seqA = tokenizer.tokenize(question)
    seqA = ["[CLS]"] + seqA + ["[SEP]"]
    t_title = tokenizer.tokenize(title)
    seqB = list()


    if passage is not None:
        seqB = seqB + t_title + ["[SEP]"] + passage + ["[SEP]"]

    truncate_input_sequence(seqA, seqB, max_seq_len)
    

    input_tokens = seqA + seqB

    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    sequence_ids = [0]*len(seqA) + [1]*len(seqB)
    input_mask = [1]*len(input_ids)
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        sequence_ids.append(0)
        input_mask.append(0)

    return (torch.LongTensor(input_ids), torch.LongTensor(input_mask), torch.LongTensor(sequence_ids))


'''
Vectorize each data point
'''

def batch_transform_bert_fever(inst, bert_max_len, bert_tokenizer):
    g = DGLGraph()

    g.add_nodes(len(inst['node']))

    question = inst['question']

    for i in range(len(inst['node'])):
        for j in range(len(inst['node'])):
            if i == j:
                continue
            g.add_edge(i, j)
    

    for i, node in enumerate(inst['node']):
        
        if node['label'] == 1:
            g.nodes[i].data['label'] = torch.tensor(1).unsqueeze(0).type(torch.FloatTensor)
        elif node['label'] ==0:
            g.nodes[i].data['label'] = torch.tensor(0).unsqueeze(0).type(torch.FloatTensor)
        
        title = node['name'].replace('_', ' ')
        context = node['context']
        encoding_inputs, encoding_masks, encoding_ids = encode_sequence_fever(question, title, context, bert_max_len, bert_tokenizer)
        g.nodes[i].data['encoding'] = encoding_inputs.unsqueeze(0)
        g.nodes[i].data['encoding_mask'] = encoding_masks.unsqueeze(0)
        g.nodes[i].data['segment_id'] = encoding_ids.unsqueeze(0)
    
    if inst['label'] == 'NOT ENOUGH INFO':
        label = 0
    elif inst['label'] == 'REFUTES':
        label = 1
    elif inst['label'] == 'SUPPORTS':
        label = 2
    else:
        print('Problem!')

    return g, inst['qid'], label



class FEVERDataset(TransformerXHDataset):
    def __init__(self, filename, config_model, isTrain=False, bert_tokenizer=None):
        super(FEVERDataset, self).__init__(filename, config_model, isTrain, bert_tokenizer)

    def __getitem__(self, index):
        index = index % self.len
        inst = self.data[index]

        return batch_transform_bert_fever(inst, self.config_model['bert_max_len'], self.bert_tokenizer)
