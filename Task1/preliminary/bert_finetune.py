import os
import json
import sys
from typing import List
import pandas as pd
import torch
import transformers
from datasets import load_dataset, Dataset
# from transformers import Trainer, TrainingArguments, TrainerState, TrainerControl, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments, TrainerState, TrainerControl, AutoModel, AutoTokenizer
import wandb
from torch import nn
from utils.parser import parse_args
from utils.parser import checkPath
import argparse
from tqdm import tqdm
"""
Unused imports:
import torch.nn as nn
import bitsandbytes as bnb
"""

## Hit@1

# <cat>
# Epoch1: 0.6492    Epoch6 : 0.8798    Epoch11: 0.9734     Epoch16: 0.9907
# Epoch2: 0.7342    Epoch7 : 0.9059    Epoch12: 0.9797     Epoch17: 0.9916
# Epoch3: 0.7813    Epoch8 : 0.9317    Epoch13: 0.9858     Epoch18: 0.9934
# Epoch4: 0.8188    Epoch9 : 0.9489    Epoch14: 0.9875     Epoch19: 0.9943
# Epoch5: 0.8512    Epoch10: 0.9630    Epoch15: 0.9892     Epoch20: 0.9950

# <sub-cat>
# Epoch1: 0.6614    Epoch6 : 0.8855    Epoch11: 0.9745     Epoch16: 0.9919
# Epoch2: 0.7337    Epoch7 : 0.9044    Epoch12: 0.9771     Epoch17: 0.9925
# Epoch3: 0.7807    Epoch8 : 0.9280    Epoch13: 0.9848     Epoch18: 0.9936
# Epoch4: 0.8143    Epoch9 : 0.9470    Epoch14: 0.9884     Epoch19: 0.9941
# Epoch5: 0.8514    Epoch10: 0.9635    Epoch15: 0.9890     Epoch20: 0.9947

# <id>
# Epoch1: 0.6528    Epoch6 : 0.8750    Epoch11: 0.9743     Epoch16: 0.9935
# Epoch2: 0.7455    Epoch7 : 0.9075    Epoch12: 0.9807     Epoch17: 0.9945
# Epoch3: 0.7864    Epoch8 : 0.9317    Epoch13: 0.9865     Epoch18: 0.9959
# Epoch4: 0.8151    Epoch9 : 0.9530    Epoch14: 0.9893     Epoch19: 0.9966
# Epoch5: 0.8508    Epoch10: 0.9631    Epoch15: 0.9924     Epoch20: 0.9970


# from peft import (
#     LoraConfig,
#     get_peft_model,
#     get_peft_model_state_dict,
#     prepare_model_for_int8_training,
#     set_peft_model_state_dict,
# )
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback

from utils.prompter import Prompter

# def add_ours_specific_args(parser):
#     parser.add_argument("--category", type=str, default='category', help="Category, Sub-category")
#     return parser

class BERT_cls(nn.Module):
    def __init__(self, args, bert_model=None):
            super(BERT_cls, self).__init__()
            self.args = args
            self.bert_model = bert_model
            self.hidden_size = bert_model.base_model.config.hidden_size
            self.id_proj = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size//2), nn.ReLU(), nn.Linear(self.hidden_size//2, len(args.id_dic_str2id)))
            self.category_proj  = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size//2), nn.ReLU(), nn.Linear(self.hidden_size//2, len(args.cat_dic_str2id))) #nn.Linear(self.hidden_size, args.goal_num)
            self.sub_category_proj = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size//2), nn.ReLU(), nn.Linear(self.hidden_size//2, len(args.sub_dic_str2id)))# nn.Linear(self.hidden_size, args.topic_num)
            # self.optimizer = torch.optim.Adam(retriever.parameters(), lr=args.lr)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs * len(train_data_loader), eta_min=args.lr * 0.1)
            # nn.init.normal_(self.goal_embedding.weight, 0, self.args.hidden_size ** -0.5)

    def forward(self, input_ids, attention_mask):
        dialog_emb = self.bert_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]  # [B, d]
        return dialog_emb

def bert_finetune(
        args,
        tokenizer,
        # evaluator,
        instructions: list = None,
        labels: list = None,
        # model/data params
        base_model: str = "bert-base-uncased",  # the only required argument
        data_path: str = "yahma/alpaca-cleaned",
        # output_dir: str = "./lora-alpaca",
        # training hyperparams
        batch_size: int = 64,
        num_epochs: int = 20,
        learning_rate: float = 1e-5,
        cutoff_len: int = 256,
        val_set_size: int = 0,

        train_on_inputs: bool = False,  # if False, masks out inputs in loss
        add_eos_token: bool = False,
        group_by_length: bool = False,  # faster, but produces an odd training loss curve
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",  # options: false | gradients | all
        wandb_log_model: str = "",  # options: false | true
        resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
        prompt_template_name: str = "alpaca_legacy",  # The prompt template to use, will default to alpaca.
):
    # output_dir = os.path.join(args.home,"model_save/bert")
    # checkPath(output_dir)
    # base_model = args.base_model
    # batch_size = args.batch_size
    # gradient_accumulation_steps = args.num_device  # update the model's weights once every gradient_accumulation_steps batches instead of updating the weights after every batch.
    # per_device_train_batch_size = batch_size // args.num_device

    # if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    #     print(
    #         f"Training Alpaca-LoRA model with params:\n"
    #         f"base_model: {base_model}\n"
    #         f"data_path: {data_path}\n"
    #         f"output_dir: {output_dir}\n"
    #         f"batch_size: {batch_size}\n"
    #         f"per_device_train_batch_size: {per_device_train_batch_size}\n"
    #         f"num_epochs: {num_epochs}\n"
    #         f"learning_rate: {learning_rate}\n"
    #         f"cutoff_len: {cutoff_len}\n"
    #         f"val_set_size: {val_set_size}\n"
    #         # f"lora_r: {lora_r}\n"
    #         # f"lora_alpha: {lora_alpha}\n"
    #         # f"lora_dropout: {lora_dropout}\n"
    #         # f"lora_target_modules: {lora_target_modules}\n"
    #         f"train_on_inputs: {train_on_inputs}\n"
    #         f"add_eos_token: {add_eos_token}\n"
    #         f"group_by_length: {group_by_length}\n"
    #         f"wandb_project: {wandb_project}\n"
    #         f"wandb_run_name: {wandb_run_name}\n"
    #         f"wandb_watch: {wandb_watch}\n"
    #         f"wandb_log_model: {wandb_log_model}\n"
    #         f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
    #         f"prompt template: {prompt_template_name}\n"
    #     )
    # assert (
    #     base_model
    # ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"
    # # gradient_accumulation_steps = batch_size // micro_batch_size

    prompter = Prompter(args, prompt_template_name)

    # device_map = "auto"

    # world_size = int(os.environ.get("WORLD_SIZE", 1))
    # print("world_size: %d" % world_size)
    # ddp = world_size != 1
    # if ddp:
    #     device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    #     gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # # Check if parameter passed or if set within environ
    # use_wandb = len(wandb_project) > 0 or (
    #         "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    # )
    # # Only overwrite environ if wandb param passed
    # if len(wandb_project) > 0:
    #     os.environ["WANDB_PROJECT"] = wandb_project
    # if len(wandb_watch) > 0:
    #     os.environ["WANDB_WATCH"] = wandb_watch
    # if len(wandb_log_model) > 0:
    #     os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    def tokenize(prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            # tokenized_full_prompt["labels"] = tokenized_full_prompt["labels"][user_prompt_len:] + [-100] * user_prompt_len  # could be sped up, probably
            
            # tokenized_full_prompt['id_lab'] = data_point["output"]
            
            tokenized_full_prompt["labels"] = tokenized_full_prompt["labels"][user_prompt_len:] + [-100] * user_prompt_len  # could be sped up, probably
            tokenized_full_prompt['cat_lab'] = data_point["output"].replace('<','>').split('>')[1].strip().lower()
            tokenized_full_prompt['sub_lab'] = data_point["output"].replace('<','>').split('>')[3].strip().lower()
            tokenized_full_prompt['id_lab'] = data_point["output"].replace('<','>').split('>')[5].strip().lower()
            
        
        return tokenized_full_prompt

    data = []
    for inst, lab in zip(instructions, labels):
        data.append({"instruction": inst, "input": "", "output": lab})
    
    if args.debug: data = data[:50]
    data = Dataset.from_pandas(pd.DataFrame(data))

    tokenizer.pad_token_id = (0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "right"  # Allow batched inference

    # if val_set_size > 0:
    #     train_val = data.train_test_split(test_size=val_set_size, shuffle=True, seed=42)
    #     train_data = (train_val["train"].shuffle().map(generate_and_tokenize_prompt))
    #     val_data = (train_val["test"].shuffle().map(generate_and_tokenize_prompt))
    # else:
    #     train_data = data.shuffle().map(generate_and_tokenize_prompt)
    #     val_data = None
    # Get-Dataset category, sub_category Key-value Dic
    
    train_data = data.shuffle().map(generate_and_tokenize_prompt)

    
    args.id_dic_str2id = {v:i for i,v in enumerate(set([i['id_lab'] for i in train_data]))}
    args.id_dic_id2str = {v:i for i,v in args.id_dic_str2id.items()}
    args.cat_dic_str2id = {v:i for i,v in enumerate(set([i['cat_lab'] for i in train_data]))}
    args.cat_dic_id2str = {v:i for i,v in args.cat_dic_str2id.items()}
    args.sub_dic_str2id = {v:i for i,v in enumerate(set([i['sub_lab'] for i in train_data]))}
    args.sub_dic_id2str = {v:i for i,v in args.sub_dic_str2id.items()}
    #
    bert_model = AutoModel.from_pretrained(base_model) # bert
    model = BERT_cls(args = args, bert_model=bert_model).to(args.device_id)

    train_dataset = BERTDataset(args, data, tokenizer)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = BERTDataset(args, data, tokenizer)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    args.num_epochs = 20
    args.lr = 1e-5
    args.device = f'{args.device_id}'
    task = args.category
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs * len(train_data_loader), eta_min=args.lr * 0.1)
    for epoch in range(20):
        model.train()
        inEpoch_BatchPlay(args, model, tokenizer, train_data_loader, optimizer, scheduler, epoch, task, mode='train')

        with torch.no_grad():
            model.eval()
            inEpoch_BatchPlay(args, model, tokenizer, test_data_loader, optimizer, scheduler, epoch, task, mode='test')

    if args.category == '': pass
    else: pass




def inEpoch_BatchPlay(args, model, tokenizer, data_loader, optimizer, scheduler, epoch, task, mode='train'):
    criterion = nn.CrossEntropyLoss().to(args.device)
    # data_loader.dataset.args.task = task
    data_loader.dataset.subtask = task

    gradient_accumulation_steps = 500
    epoch_loss, steps = 0, 0

    torch.cuda.empty_cache()
    contexts, labels, cats, subs, ids, preds = [],[],[],[],[],[]
    for batch in tqdm(data_loader, desc=f"Epoch_{epoch}_{task:^5}_{mode:^5}", bar_format=' {l_bar} | {bar:23} {r_bar}'):
        # input_ids, attention_mask, response, goal_idx, topic_idx = [batch[i].to(args.device) for i in ["input_ids", "attention_mask", "response", 'goal_idx', 'topic_idx']]
        input_ids, attention_mask, output, cat_idx, sub_idx, id_idx = [batch[i].to(args.device) for i in ['input_ids', 'attention_mask', 'output', 'cat_idx', 'sub_idx', 'id_idx']]
        # target = goal_idx if task == 'goal' else topic_idx
        # Model Forwarding
        dialog_emb = model(input_ids=input_ids, attention_mask=attention_mask)  # [B, d]
        scores = 0.0
        if 'id' in task:
            scores = model.id_category_proj(dialob_emb)
            loss = criterion(scores, id_idx)
        elif 'sub' in task:
            scores = model.sub_category_proj(dialog_emb)
            loss = criterion(scores, sub_idx)
        else:
            scores = model.category_proj(dialog_emb)
            loss = criterion(scores, cat_idx)
        epoch_loss += loss

        if 'train' == mode:
            optimizer.zero_grad()
            loss.backward()
            if (steps + 1) % gradient_accumulation_steps == 0: torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            loss.detach()
            model.zero_grad()
        if 'test' == mode:
            batch_top_results = torch.topk(scores, dim=-1, k=1).indices
            # if 'sub' in task: results = [args.sub_dic_id2str[i] for i in batch_top_results]
            # else: results = [args.cat_dic_id2str[i] for i in batch_top_results]
            preds.extend(batch_top_results)
            ids.extend([int(i) for i in id_idx.data])
            cats.extend([int(i) for i in cat_idx.data])
            subs.extend([int(i) for i in sub_idx.data])
            # for i in zip(batch_top_results):
            #     ground = 
    if 'test' == mode:
        if 'id' in task:
            ground_truth = ids
            generated_results = [f"gen: {args.id_dic_id2str[int(p)]} | lab: {args.id_dic_id2str[q]}" for p,q in zip(preds, ground_truth)]
        elif 'sub' in task:
            ground_truth = subs
            generated_results = [f"gen: {args.sub_dic_id2str[int(p)]} | lab: {args.sub_dic_id2str[q]}" for p,q in zip(preds, ground_truth)]
        else:
            ground_truth = cats
            generated_results = [f"gen: {args.cat_dic_id2str[int(p)]} | lab: {args.cat_dic_id2str[q]}" for p,q in zip(preds, ground_truth)]
        if args.write:
            for i in generated_results:
                args.log_file.write(json.dumps(i, ensure_ascii=False) + '\n')
        results = [int(p)==g for p,g in zip(preds, ground_truth)] 
        hit_score =  sum(results) / len(results)
        print(f'Hit@1: {hit_score:.4f}')
        print(f'Total Count: {len(preds):.4f}')
        # elif 'test' == mode:
        #     if 'sub' in task:
        #         sub_word = args.sub_dic_id2str[]
        #     else:
        #         cat_word = args.cat_dic_id2str[]

    #     scores = torch.softmax(scores, dim=-1)
    #     if task=='goal' and args.version == 'ko':  topk=1
    #     else: topk=5
    #     topk_pred = [list(i) for i in torch.topk(scores, k=topk, dim=-1).indices.detach().cpu().numpy()]
    #     topk_conf = [list(i) for i in torch.topk(scores, k=topk, dim=-1).values.detach().cpu().numpy()]
    #     ## For Scoring and Print
    #     contexts.extend(tokenizer.batch_decode(input_ids))
    #     task_preds.extend(topk_pred)
    #     task_confs.extend(topk_conf)
    #     task_labels.extend([int(i) for i in target.detach()])
    #     gold_goal.extend([int(i) for i in goal_idx])
    #     gold_topic.extend([int(i) for i in topic_idx])

    #     # if task=='topic' and mode=='test': predicted_goal_True_cnt.extend([real_goal==pred_goal for real_goal, pred_goal  in zip(goal_idx, batch['predicted_goal_idx'])])

    # hit1_ratio = sum([label == preds[0] for preds, label in zip(task_preds, task_labels)]) / len(task_preds)

    # Hitdic, Hitdic_ratio, output_str = HitbyType(args, task_preds, task_labels, gold_goal)
    # assert Hitdic['Total']['total'] == len(data_loader.dataset)
    # if mode == 'test':
    #     for i in output_str:
    #         logger.info(f"{mode}_{epoch}_{task} {i}")
    # if 'train' == mode: scheduler.step()
    # savePrint(args, contexts, task_preds, task_labels, gold_goal, gold_topic, epoch, task, mode)
    torch.cuda.empty_cache()
    # return task_preds, task_confs, hit1_ratio




if __name__ == "__main__":
    # fire.Fire(llama_finetune)
    parser = argparse.ArgumentParser(description="ours_main.py")
    # parser = add_ours_specific_args(parser)

    args = parse_args(parser)
    bert_finetune(args, num_epochs=args.epoch)




from collections import defaultdict
from copy import deepcopy
import random
class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, args, augmented_raw_sample, tokenizer=None, mode='train'):
        super(BERTDataset, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.augmented_raw_sample = augmented_raw_sample
        self.mode = mode
        self.input_max_length = 256
        self.target_max_length = 256
        # self.knowledgeDB = knowledgeDB

    def set_tokenizer(self, tokenizer): self.tokenizer = tokenizer

    def __getitem__(self, item):
        data = self.augmented_raw_sample[item]
        # {"instruction": inst, "input": "", "output": lab}
        cbdicKeys = ['instruction', 'input', 'output']
        instruction, input, output = [data[i] for i in cbdicKeys]

        pad_token_id = self.tokenizer.pad_token_id

        context_batch = defaultdict()
        
        input_sent = self.tokenizer(instruction, padding='max_length', truncation=True, max_length=self.input_max_length)

        # input_sentence = self.tokenizer.question_encoder('<dialog>' + dialog, add_special_tokens=False).input_ids
        # input_sentence = [self.tokenizer.question_encoder.cls_token_id] + prefix_encoding + input_sentence[-(self.input_max_length - len(prefix_encoding) - 1):]
        # input_sentence = input_sentence + [pad_token_id] * (self.input_max_length - len(input_sentence))

        context_batch['input_ids'] = torch.LongTensor(input_sent.input_ids).to(self.args.device)
        attention_mask = context_batch['input_ids'].ne(pad_token_id)
        context_batch['attention_mask'] = attention_mask
        # response에서 [SEP] token 제거

        # if '[SEP]' in response: response = response[: response.index("[SEP]")]

        labels = self.tokenizer(output, max_length=self.target_max_length, padding='max_length', truncation=True)['input_ids']
        context_batch['output'] = labels

        
        cat = output.replace('<','>').split('>')[1].strip().lower()
        sub = output.replace('<','>').split('>')[3].strip().lower()
        id = output.replace('<','>').split('>')[5].strip().lower()
        context_batch['cat_idx'] = self.args.cat_dic_str2id[cat]  # index로 바꿈
        context_batch['sub_idx'] = self.args.sub_dic_str2id[sub]  # index로 바꿈
        context_batch['id_idx'] = self.args.id_dic_str2id[id]  # index로 바꿈
        for k, v in context_batch.items():
            if not isinstance(v, torch.Tensor):
                context_batch[k] = torch.as_tensor(v, device=self.args.device)
        return context_batch
# input_ids, attention_mask, output, cat_idx, sub_idx
    def __len__(self):
        return len(self.augmented_raw_sample)