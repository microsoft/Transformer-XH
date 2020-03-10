# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# specific stuff for hotpot qa
import torch
from torch.utils.data import Dataset, DataLoader
import dgl
from dgl import DGLGraph
import dgl.function as fn
from .base import TransformerXHDataset
from .utils import truncate_input_sequence


def batcher_hotpot(device):
    def batcher_dev(batch):

        graph, qid = batch[0]
        batch_graphs = dgl.batch([batch[0][0]])
        batch_graphs.ndata['encoding'] = batch_graphs.ndata['encoding'].to(device)
        batch_graphs.ndata['encoding_mask'] =batch_graphs.ndata['encoding_mask'].to(device)
        batch_graphs.ndata['segment_id'] =batch_graphs.ndata['segment_id'].to(device)
        qid = [batch[0][1]]

        return batch_graphs, batch_graphs.ndata['label'].to(device), batch_graphs.ndata['label_start'].to(device), batch_graphs.ndata['label_end'].to(device), batch_graphs.ndata['span_label'].to(device), qid
    return batcher_dev


def encode_sequence_hotpot(question, passage, evidence, max_seq_len, tokenizer):
    
    seqA = tokenizer.tokenize(question)
    seqA = ["[CLS]"] + seqA + ["[SEP]"]
    seqB = list()
    for evi in evidence:
        seq_tokens = tokenizer.tokenize(evi)
        seqA = seqA + seq_tokens
        seqA = seqA + ["[SEP]"]
    if passage is not None:
        seqB = seqB + passage + ["[SEP]"]
    truncate_input_sequence(seqA, seqB, max_seq_len)
    
    input_tokens = seqA + seqB

    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    sequence_ids = [0]*len(seqA) + [1]*len(seqB)
    input_mask = [1]*len(input_ids)
    B_start = [len(seqA)]
    while len(input_ids) < max_seq_len:
        input_ids.append(0)
        sequence_ids.append(0)
        input_mask.append(0)

    return (torch.LongTensor(input_ids), torch.LongTensor(input_mask), torch.LongTensor(sequence_ids),  torch.LongTensor(B_start))



'''
Vectorize each data point
'''

def batch_transform_bert_hotpot(inst, bert_max_len, bert_tokenizer):
    g = DGLGraph()

    g.add_nodes(len(inst['node']))

    question = inst['question']


    for i, node in enumerate(inst['node']):
        inst['node'][i]['evidence'] = list()

    #### concatenate all edge sentences

    edge_dict = dict()
    for edge in inst['edge']:
        e_start = edge['start']
        e_end = edge['end']

        idx = (e_start, e_end)
        if idx not in edge_dict:
            edge_dict[idx] =list()
        if edge['sent'] not in edge_dict[idx]:
            edge_dict[idx].append(edge['sent'])
    
    for idx, context in edge_dict.items():
        start, end = idx
        g.add_edge(start, end)
        for sent in context:
            inst['node'][end]['evidence'].append(sent)

    for i, node in enumerate(inst['node']):

        context = node['context']
        evidence = list(set(node['evidence']))

        encoding_inputs, encoding_masks, encoding_ids, B_start = encode_sequence_hotpot(question, context, evidence, bert_max_len, bert_tokenizer)
        g.nodes[i].data['encoding'] = encoding_inputs.unsqueeze(0)
        g.nodes[i].data['encoding_mask'] = encoding_masks.unsqueeze(0)
        g.nodes[i].data['segment_id'] = encoding_ids.unsqueeze(0)
        g.nodes[i].data['B_start'] = B_start

        if node['is_ans'] == 1:
            g.nodes[i].data['label'] = torch.tensor(1).unsqueeze(0).type(torch.FloatTensor)
        elif node['is_ans'] ==0:
            g.nodes[i].data['label'] = torch.tensor(0).unsqueeze(0).type(torch.FloatTensor)
        else:
            g.nodes[i].data['label'] = torch.tensor(-1).unsqueeze(0).type(torch.FloatTensor)

        spans = node['spans']
        if node['is_ans'] != 1:
            g.nodes[i].data['span_label'] = torch.tensor(-1).unsqueeze(0).type(torch.FloatTensor)
        else:
            g.nodes[i].data['span_label'] = torch.tensor(0).unsqueeze(0).type(torch.FloatTensor)

        if len(spans) == 0:
            spans = [(-1, -1)]
        start_spans = torch.LongTensor([p[0] for p in spans])
        end_spans = torch.LongTensor([p[1] for p in spans])
        start_spans = start_spans + B_start
        end_spans = end_spans + B_start

        g.nodes[i].data['label_start'] = start_spans
        g.nodes[i].data['label_end'] = end_spans

    return g, inst['qid']






class HotpotDataset(TransformerXHDataset):
    def __init__(self, filename, config_model, isTrain=False, bert_tokenizer=None):
        super(HotpotDataset, self).__init__(filename, config_model, isTrain, bert_tokenizer)


    def __getitem__(self, index):
        index = index % self.len
        inst = self.data[index]

        return batch_transform_bert_hotpot(inst, self.config_model['bert_max_len'], self.bert_tokenizer)


