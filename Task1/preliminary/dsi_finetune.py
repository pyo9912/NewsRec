import argparse
import os
import sys
import json
import torch
from torch import nn
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, PreTrainedTokenizer, DataCollatorWithPadding, TrainerCallback, AutoConfig, AutoModel, AutoTokenizer
from transformers.trainer import Trainer
from typing import Dict, List, Tuple, Optional, Any, Union
# from kmeans_pytorch import kmeans

import wandb
from tqdm import tqdm
from datetime import datetime
from pytz import timezone

from utils.parser import checkPath
from utils.prompter import Prompter

class IndexingCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [x[0] for x in features]
        newsids = [x[1] for x in features]
        inputs = self.tokenizer(
            input_ids, padding="max_length", max_length=32, truncation=True, return_tensors="pt"
        )
        
        labels = self.tokenizer(
            newsids, padding="longest", return_tensors="pt"
        ).input_ids

        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        return inputs

class IndexingTrainer(Trainer):
    def __init__(self, restrict_decode_vocab, **kwds):
        super().__init__(**kwds)
        self.restrict_decode_vocab = restrict_decode_vocab

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels']).loss
        if return_outputs:
            return loss, [None, None]  # fake outputs
        return loss

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()
        print("==============================prediction step==============================")
        # eval_loss = super().prediction_step(model, inputs, True, ignore_keys)[0]
        with torch.no_grad():
            # greedy search
            ans_ids = model.generate(
                inputs['input_ids'].to(self.args.device),
                max_length=32,
                prefix_allowed_tokens_fn=self.restrict_decode_vocab,
                early_stopping=True,)
        return (None, ans_ids, inputs['labels'])


def compute_metrics(eval_preds):
    num_predict = 0
    num_correct = 0
    for predict, label in zip(eval_preds.predictions, eval_preds.label_ids):
        num_predict += 1
        if len(np.where(predict == 1)[0]) == 0:
            continue
        if np.array_equal(label[:np.where(label == 1)[0].item()],
                          predict[np.where(predict == 0)[0][0].item() + 1:np.where(predict == 1)[0].item()]):
            num_correct += 1

    return {'accuracy': num_correct / num_predict}


def dsi_finetune(
        args,
        tokenizer,
        restrict_decode_vocab,
        instructions: list = None,
        labels: list = None,
        # model/data params
        base_model: str = "",
        data_path: str = "",
        output_dir: str = "",
        cache_dir: str = "",
        # training hyperparams
        batch_size: int = 64,
        num_epochs: int = 30,
        learning_rate: float = 1e-4,
        cutoff_len: int = 128,
        val_set_size: int = 0,
        # llm hyperparmas
        train_on_inputs: bool = False,
        add_eos_token: bool = False,
        group_by_length: bool = False,
        # wandb params
        wandb_project: str = "",
        wandb_run_name: str = "",
        wandb_watch: str = "",
        wandb_log_model: str = "",
        resume_from_checkpoint: str = None,
        prompt_template_name: str = "alpaca_legacy",
):
    restrict_decode_vocab = restrict_decode_vocab
    output_dir = os.path.join(args.home,"model_save/DSI")
    checkPath(output_dir)
    cache_dir = os.path.join(args.home,"cache_save/DSI")
    checkPath(cache_dir)
    base_model = args.base_model
    batch_size = args.batch_size
    gradient_accumulation_steps = args.num_device
    per_device_train_batch_size = batch_size // args.num_device

    prompter = Prompter(args, prompt_template_name)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model


    data = []
    for inst, lab in zip(instructions, labels):
        data.append((inst,lab))
    if args.debug: data = data[:100]
    sample_num = int(args.sample_num)
    data = data[:sample_num]
    # data = Dataset.from_pandas(pd.DataFrame(data))

    if val_set_size > 0:
        train_val = data.train_test_split(
            test_size=val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"]#.shuffle(generate_and_tokenize_prompt)
        )
        val_data = (
            train_val["test"]#.shuffle(generate_and_tokenize_prompt)
        )
    else:
        # train_data = data.shuffle(generate_and_tokenize_prompt)
        train_data = data
        val_data = None
    
    print("=================================")
    print("LEN TRAIN DATASET: ", len(train_data))
    print("=================================\n")
    

    ################################################################
    # idx = 0
    # batch_size = 32
    # start = idx
    # last = min(idx+ batch_size, num_news)
    # train_dataset[idx:idx+batch_size]
    # idx += batch_size

    # bert_model = AutoModel.from_pretrained('bert-base-uncased')
    # bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    # title_emb_list = []
    # for idx, data in tqdm(enumerate(train_dataset)): # todo: batchify
    #     title_text = data[0]
    #     news_id = data[1]
    #     title_text_token = bert_tokenizer(title_text, return_tensors='pt')
    #     title_text_emb = bert_model(input_ids=title_text_token.input_ids).last_hidden_state[:,0,:]
    #     title_emb_list.append(title_text_emb.squeeze(dim=0))
    # title_emb_list = torch.stack(title_emb_list, dim=0)
    # cluster_ids, cluster_centroids = kmeans(title_emb_list, 10, device=torch.device(f"cuda:{args.device_id}"))
    

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,  # 0.0005,
        warmup_steps=100,#len(train_data) / 60,
        # weight_decay=0.01,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        # evaluation_strategy=args.evaluation_strategy,
        # eval_steps=1000,
        dataloader_drop_last=False,  # necessary
        report_to='wandb',
        logging_steps=50,
        save_strategy='no',
        num_train_epochs=num_epochs,
        lr_scheduler_type="constant_with_warmup",
        # fp16=True,  # gives 0/nan loss at some point during training, seems this is a transformers bug.
        dataloader_num_workers=0 #1
        # gradient_accumulation_steps=2
    )
    saved_model_path = output_dir
    if not os.path.exists(saved_model_path) or args.write:
        print("Write model")
        checkPath(saved_model_path)
        model = T5ForConditionalGeneration.from_pretrained(
            base_model,
            cache_dir=cache_dir,
        )
        print("=================================")
        print("INDEXING TASK CHECKING")
        print("LEN INDEX DATASET: ", len(train_data))
        print("=================================\n")
        
        index_trainer = IndexingTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_data,
            # eval_dataset=eval_dataset,
            data_collator=IndexingCollator(
                tokenizer,
                padding='longest',
            ),
            compute_metrics=compute_metrics,
            restrict_decode_vocab=restrict_decode_vocab
        )
        print(index_trainer.train_dataset[0])
        print("=============Train indexing=============")
        index_trainer.train()
        index_trainer.save_model(saved_model_path)
    

    else:
        print("call model")
        model = T5ForConditionalGeneration.from_pretrained(saved_model_path)



# if __name__ == "__main__":
#     args = parse_args()
#     main(args)
