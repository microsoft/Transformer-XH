# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
import dgl.function as fn

from pytorch_transformers import *
from pytorch_transformers.modeling_bert import BertModel, BertEncoder, BertPreTrainedModel
from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE


'''
Graph Attention network component
'''


class GraphAttention(nn.Module):
    def __init__(self, in_dim=64,
                 out_dim=64,
                 num_heads=12,
                 feat_drop=0.6,
                 attn_drop=0.6,
                 alpha=0.2,
                 residual=True):

        super(GraphAttention, self).__init__()
        self.num_heads = num_heads
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x : x
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x : x
        self.attn_l = nn.Parameter(torch.Tensor(size=(1, num_heads, out_dim)))
        self.attn_r = nn.Parameter(torch.Tensor(size=(1, num_heads, out_dim)))
        nn.init.xavier_normal_(self.fc.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_l.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_r.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.residual = residual
        self.activation = nn.ReLU()
        self.res_fc = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_normal_(self.res_fc.weight.data, gain=1.414)

    ### this is Gragh attention network part, we follow standard inplementation from DGL library
    def forward(self, g):
        self.g = g
        h = g.ndata['h']
        h = h.reshape((h.shape[0], self.num_heads, -1))
        ft = self.fc(h) 
        a1 = (ft * self.attn_l).sum(dim=-1).unsqueeze(-1) 
        a2 = (ft * self.attn_r).sum(dim=-1).unsqueeze(-1) 
        g.ndata.update({'ft' : ft, 'a1' : a1, 'a2' : a2})
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        ret = g.ndata['ft']
        if self.residual:
            if self.res_fc is not None:
                resval = self.res_fc(h)  
            else:
                resval = torch.unsqueeze(h, 1) 
            ret = resval + ret
        g.ndata['h'] = self.activation(ret.flatten(1))
        

    def message_func(self, edges):
        return {'z': edges.src['ft'], 'a': edges.data['a']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['a'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'ft': h}


    def edge_attention(self, edges):
        a = self.leaky_relu(edges.src['a1'] + edges.dst['a2'])
        return {'a' : a}





'''
Transformer-XH Encoder, we apply on last three BERT layers 

'''

class TransformerXHEncoder(BertEncoder):
    def __init__(self, config):
        super(TransformerXHEncoder, self).__init__(config)
        self.heads = ([8] * 1) + [1]
        self.config = config
        self.build_model()
        ### Here we apply on the last three layers, but it's ok to try different layers here.
        self.linear_layer1 =  nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.linear_layer2 =  nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.linear_layer3 =  nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.linear_layer1.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.linear_layer2.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.linear_layer3.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        
    
    def build_model(self):
        self.graph_layers = nn.ModuleList()
        # input to hidden
        device = torch.device("cuda")

        i2h = self.build_input_layer().to(device)
        self.graph_layers.append(i2h)
        # hidden to hidden
        h2h = self.build_hidden_layer().to(device)
        self.graph_layers.append(h2h)
        h2h = self.build_hidden_layer().to(device)
        self.graph_layers.append(h2h)

    ### here the graph has dimension 64, with 12 heads, the dropout rates are 0.6
    def build_input_layer(self):
        return GraphAttention()

    def build_hidden_layer(self):
        return GraphAttention()

    def forward(self, graph, hidden_states, attention_mask, gnn_layer_num, output_all_encoded_layers=True):
        all_encoder_layers = []
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)[0]
            pooled_output = hidden_states[:, 0]
            graph.ndata['h'] = pooled_output
            
            if i >= gnn_layer_num:
                if i == 9:
                    g_layer = self.graph_layers[0]
                    g_layer(graph)
                    graph_outputs = graph.ndata.pop('h')
                    ht_ori = hidden_states.clone()
                    ht_ori[:, 0] =  self.linear_layer1(torch.cat((graph_outputs, pooled_output), -1))
                elif i == 10:
                    g_layer = self.graph_layers[1]
                    g_layer(graph)
                    graph_outputs = graph.ndata.pop('h')
                    ht_ori = hidden_states.clone()
                    ht_ori[:, 0] =  self.linear_layer2(torch.cat((graph_outputs, pooled_output), -1))
                else:
                    g_layer = self.graph_layers[2]
                    g_layer(graph)
                    graph_outputs = graph.ndata.pop('h')
                    ht_ori = hidden_states.clone()
                    ht_ori[:, 0] =  self.linear_layer3(torch.cat((graph_outputs, pooled_output), -1))
                hidden_states = ht_ori
                if output_all_encoded_layers:
                    all_encoder_layers.append(ht_ori)
            else:
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


'''
Transformer-XH main class
'''


class Transformer_xh(BertModel):
    def __init__(self, config):
        super(Transformer_xh, self).__init__(config)

        self.encoder = TransformerXHEncoder(config)
    
    def forward(self, graph, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, gnn_layer=11):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(graph, embedding_output,
                                       extended_attention_mask, gnn_layer)
        sequence_output = encoder_outputs[-1]
        pooled_output = self.pooler(sequence_output)
        outputs = sequence_output, pooled_output  # add hidden_states and attentions if they are here

        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


'''
Model network component, it is inherited by different tasks (at model_fever.py and model_hotpot.py file).
The representation layers are Transformer-XH and the last layer is task specific.
'''

class ModelHelper(BertPreTrainedModel):
    def __init__(self, node_encoder: BertModel, args, bert_config, config_model):
        super(ModelHelper, self).__init__(bert_config)
        ### node_encoder -> Transformer-XH
        self.node_encoder = node_encoder
        self.config_model = config_model
        self.args = args
        self.node_dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.final_layer = nn.Linear(self.config.hidden_size, 1)
        self.final_layer.apply(self.init_weights)

    def forward(self, batch, device):
        pass
        

'''
Model Wrapper
'''

class Model:
    def __init__(self, args, config):
        self.config = config
        self.config_model = config['model']
        self.args = args
        self.bert_node_encoder = Transformer_xh.from_pretrained(self.config['bert_model_file'], cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))
        self.bert_config = self.bert_node_encoder.config
        self.network= ModelHelper(self.bert_node_encoder, self.args, self.bert_config, self.config_model)
        self.device= args.device

    def half(self):
        self.network.half()
    
    def eval(self):
        self.network.eval()

    def train(self):
        self.network.train()
    
    def save(self, filename: str):
        network = self.network
        if isinstance(network, nn.DataParallel):
            network = network.module

        return torch.save(self.network.state_dict(), filename)

    def load(self, model_state_dict: str):
        return self.network.load_state_dict(torch.load(model_state_dict, map_location=lambda storage, loc: storage))

    
