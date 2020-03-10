# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .model import Model, ModelHelper
import torch
import torch.nn as nn


class ModelHelper_Hotpot(ModelHelper):
    def __init__(self, node_encoder, args, bert_config, config_model):
        super(ModelHelper_Hotpot, self).__init__(node_encoder, args, bert_config, config_model)
        self.mrc_final_layer = nn.Linear(self.config.hidden_size, 2)
        self.mrc_final_layer.apply(self.init_weights)
    
    
    def forward(self, batch, device):    
        ### Transformer-XH for node representations
        g = batch[0]
        g.ndata['encoding'] = g.ndata['encoding'].to(device)
        g.ndata['encoding_mask'] = g.ndata['encoding_mask'].to(device)
        g.ndata['segment_id'] = g.ndata['segment_id'].to(device)
        outputs = self.node_encoder(g, g.ndata['encoding'], g.ndata['segment_id'], g.ndata['encoding_mask'], gnn_layer=self.config_model['gnn_layer'])
        node_sequence_output = outputs[0]
        node_pooled_output = outputs[1]
        node_pooled_output = self.node_dropout(node_pooled_output)
        
        #### Task specific layer (last layer)
        mrc_logits = self.mrc_final_layer(node_sequence_output)
        logits = self.final_layer(node_pooled_output).squeeze(-1)

        return logits, mrc_logits




class Model_Hotpot(Model):
    def __init__(self, args, config):
        super(Model_Hotpot, self).__init__(args, config)
        self.network= ModelHelper_Hotpot(self.bert_node_encoder, self.args, self.bert_config, self.config_model)




