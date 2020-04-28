# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from data import HotpotDataset, FEVERDataset, TransformerXHDataset

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from data import batcher_hotpot, batcher_fever
import torch
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
from tqdm import tqdm
import os
import logging
import torch.nn.functional as F
from Evaluator import evaluation_hotpot, evaluation_fever



'''
Training for Hotpot QA task
'''

def train_hotpot(model, index, config, args, best_score, optimizer, scheduler):
    model.train()
    dataset = HotpotDataset(config["system"]['train_data'], config["model"], True, args.tokenizer)
    device = args.device
    train_sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset=dataset,
                              sampler = train_sampler,
                              batch_size=config['training']['train_batch_size'],
                              collate_fn=batcher_hotpot(device),
                              num_workers=0)
                            
    
    print_loss = 0 
    bce_loss_logits = nn.BCEWithLogitsLoss()

    for step, batch in enumerate(tqdm(dataloader)):

        logits, mrc_logits = model.network(batch, device)
        pos_node_idx = [i for i in range(batch[1].size(0)) if batch[1][i].item() != -1]
        if args.fp16:
            node_loss = bce_loss_logits(logits[pos_node_idx], batch[1][pos_node_idx].half())
        else:
            node_loss = bce_loss_logits(logits[pos_node_idx], batch[1][pos_node_idx])
        
        start_logits, end_logits = mrc_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        pos_idx = [i for i in range(batch[4].size(0)) if batch[4][i].item() == 0]
        start_positions = batch[2]
        end_positions = batch[3]

        # sometimes the start/end positions are outside our model inputs, we ignore these exs
        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits[pos_idx], start_positions[pos_idx])
        end_loss = loss_fct(end_logits[pos_idx], end_positions[pos_idx])

        loss = (start_loss + end_loss) / 2 + node_loss


        if args.n_gpu > 1:
            loss = loss.mean()

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        print_loss +=  loss.data.cpu().numpy()
                
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        if (step + 1) % args.checkpoint == 0:
            logging.info("********* loss ************{}".format(print_loss))
            print_loss = 0
            model.eval()
            eval_file = config['system']['validation_data']
            auc, _ = evaluation_hotpot(model, eval_file, config, args)
            if auc > best_score:
                best_score = auc
                model.save(os.path.join(base_dir, config['name'], "saved_models/model_finetuned_epoch_{}.pt".format(0)))
        
        model.train()
        
    return best_score


'''
Training for FEVER task
'''

def train_fever(model, index, config, args, best_score, optimizer, scheduler):
    model.train()
    dataset = FEVERDataset(config["system"]['train_data'], config["model"], True, args.tokenizer)
    device = args.device
    train_sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset=dataset,
                              sampler = train_sampler,
                              batch_size=config['training']['train_batch_size'],
                              collate_fn=batcher_fever(device),
                              num_workers=0)
                            
    
    print_loss = 0 
    criterion = CrossEntropyLoss()
    bce_loss_logits = nn.BCEWithLogitsLoss()
    for step, batch in enumerate(tqdm(dataloader)):

        logits_score, logits_pred = model.network(batch, device)

        if args.fp16:
            node_loss = bce_loss_logits(logits_score, batch[1].half())
        else:
            node_loss = bce_loss_logits(logits_score, batch[1])

        logits_score = F.softmax(logits_score)
        logits_pred =  F.softmax(logits_pred, dim=1)
        final_score = torch.mm(logits_score.unsqueeze(0), logits_pred)
        pred_loss = criterion(final_score, batch[2])
        

        loss = pred_loss + node_loss
        if args.n_gpu > 1:
            loss = loss.mean()

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        print_loss +=  loss.data.cpu().numpy()
                
        if args.fp16:
            optimizer.backward(loss)
        else:
            loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
                
        if (step + 1) % args.checkpoint == 0:    
            logging.info("********* loss ************{}".format(print_loss))
            print_loss = 0
            model.eval()
            eval_file = config['system']['validation_data']
            auc, _ = evaluation_fever(model, eval_file, config, args)
            if auc > best_score:
                best_score = auc
                model.save(os.path.join(base_dir, config['name'], "saved_models/model_finetuned_epoch_{}.pt".format(0)))
        
        model.train()
        
    return best_score

