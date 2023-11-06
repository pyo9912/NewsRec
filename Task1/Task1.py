import os

import wandb
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, T5Tokenizer
import transformers
import torch
import json
import argparse
import logging
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from pytz import timezone

# from chatgpt_test import chatgpt_test

from utils.data import read_data
from utils.parser import parse_args
from utils.parser import checkPath


class RQ(Dataset):
    def __init__(self, tokenizer, args):
        super(Dataset, self).__init__()
        self.data_samples = []
        self.tokenizer = tokenizer
        self.args = args
        self.read_data()

    def read_data(self):
        RQ_data = json.load((open('data/rq' + str(self.args.rq_num) + '.json', 'r', encoding='utf-8')))
        question, answer = [], []
        for data in RQ_data:
            question.append(data['Question'])
            answer.append(data['Answer'])

        # tokenized_input = self.tokenizer(question, return_tensors="pt", padding=True, return_token_type_ids=False).to(
        #     self.args.device_id)
        # tokenized_output = self.tokenizer(answer, return_tensors="pt", padding=True, return_token_type_ids=False).to(
        #     self.args.device_id)
        for t_input, t_output in zip(question, answer):
            self.data_samples.append((t_input, t_output))

    def __getitem__(self, idx):
        input = self.data_samples[idx][0]
        output = self.data_samples[idx][1]

        return input, output

    def __len__(self):
        return len(self.data_samples)


class RQCollator:
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, data_batch):
        question_batch, resp_batch, input_len_batch = [], [], []
        for data_input, data_output in data_batch:
            question_batch.append(data_input)
            input_len_batch.append(len(data_input))
            resp_batch.append(data_output)

        input_batch = {}
        tokenized_input = self.tokenizer(question_batch, return_tensors="pt", padding=True,
                                         return_token_type_ids=False).to(
            self.args.device_id)
        input_batch['answer'] = resp_batch
        input_batch['question_len'] = torch.sum(tokenized_input.attention_mask, dim=1)
        input_batch['question'] = tokenized_input

        return input_batch


def evaluate(gen_seq, answer, log_file):
    # gen_output, result_f = [], []
    # for seq, len in zip(gen_seq, input_len):
    #     gen_output.append(seq[len:])
    # decoded_output = tokenizer.batch_decode(gen_output, skip_special_tokens=True)
    for output, label in zip(gen_seq, answer):
        log_file.write(json.dumps({'GEN': output, 'ANSWER': label}, ensure_ascii=False) + '\n')
    #     result_f.append({'GEN': output, 'ANSWER': label})
    # with open('result/llama/' + str(rq_num) + '_result.json', 'w', encoding='utf-8') as f_write:
    #     f_write.write(json.dumps(result_f, indent=4))


if __name__ == '__main__':
    args = parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"]=args.device_id
    
    mdhm = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))
    result_path = os.path.join(args.output_dir, args.base_model.replace('/', '-'))
    if not os.path.exists(result_path): os.mkdir(result_path)

    if args.log_file == '':
        log_file = open(os.path.join(result_path, f'rq{args.rq_num}_{mdhm}.json'), 'a', buffering=1, encoding='UTF-8')
    else:
        log_file = open(os.path.join(result_path, f'{args.log_file}.json'), 'a', buffering=1, encoding='UTF-8')

    args.log_file = log_file
    question_data = read_data(args)
    instructions = [i[0] for i in question_data]
    labels = [i[1] for i in question_data]

    # print("Instructions: ",instructions[0]) # Following text is an ID of the news. 'N88753'. what is the title of the news?
    # print("labels", labels[0]) # The Brands Queen Elizabeth, Prince Charles, and Prince Philip Swear By

    wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    # if 'gpt' in args.base_model.lower():
    #     chatgpt_test(args=args, instructions=instructions, labels=labels)
    
    if 'llama' in args.base_model.lower(): # llama
        from preliminary.llama_finetune import llama_finetune
        from preliminary.llama_test import LLaMaEvaluator
        tokenizer = LlamaTokenizer.from_pretrained(args.base_model)
        evaluator = LLaMaEvaluator(args=args, tokenizer=tokenizer, restrict_decode_vocab=None, instructions=instructions, labels=labels)

        if 'train' in args.mode:
            llama_finetune(args=args, evaluator=evaluator, tokenizer=tokenizer, instructions=instructions, labels=labels)
            # evaluator.test()
        if 'test' == args.mode:
            evaluator.test()

    elif 'google' in args.base_model.lower():  # google-T5
        from preliminary.t5_finetune import t5_finetune
        from preliminary.t5_test import T5Evaluator
        tokenizer = T5Tokenizer.from_pretrained(args.base_model)
        evaluator = T5Evaluator(args=args, tokenizer=tokenizer, instructions=instructions, labels=labels)
        
        if 'train' in args.mode:
            # llama_finetune(args=args, evaluator=evaluator, tokenizer=tokenizer, instructions=instructions, labels=labels)
            t5_finetune(args=args, evaluator=evaluator, tokenizer=tokenizer, instructions=instructions, labels=labels)
            # evaluator.test()
        if 'test' == args.mode:
            evaluator.test()

    elif 'bert' in args.base_model.lower():  # BERT
        from preliminary.bert_finetune import bert_finetune
        # if args.debug: args.device_id='cpu'
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        # evaluator = BERTEvaluator(args=args, tokenizer=tokenizer, instructions=instructions, labels=labels)
        
        if 'train' in args.mode:
            # llama_finetune(args=args, evaluator=evaluator, tokenizer=tokenizer, instructions=instructions, labels=labels)
            bert_finetune(args=args, tokenizer=tokenizer, instructions=instructions, labels=labels)
            # evaluator.test()
        # if 'test' == args.mode:
            # evaluator.test()

    elif 't5' in args.base_model.lower(): # T5-dsi
        from preliminary.dsi_finetune import dsi_finetune
        from preliminary.dsi_test import DsiEvaluator

        cache_dir = os.path.join(args.home, "cache_save/DSI")
        checkPath(cache_dir)

        tokenizer = T5Tokenizer.from_pretrained(args.base_model,cache_dir=cache_dir)
        evaluator = DsiEvaluator(args=args, tokenizer=tokenizer, instructions=instructions, labels=labels)

        ### Restrict decode vocab
        SPIECE_UNDERLINE = "▁"
        INT_TOKEN_IDS = []
        for token, id in tokenizer.get_vocab().items():
            if token[0] == SPIECE_UNDERLINE:
                if token[1:].isdigit():
                    INT_TOKEN_IDS.append(id)
            if token == SPIECE_UNDERLINE:
                INT_TOKEN_IDS.append(id)
            elif token.isdigit():
                INT_TOKEN_IDS.append(id)
        INT_TOKEN_IDS.append(tokenizer.eos_token_id)


        def restrict_decode_vocab(batch_idx, prefix_beam):
            return INT_TOKEN_IDS
        ###

        if 'train' in args.mode:
            dsi_finetune(args=args, tokenizer=tokenizer, restrict_decode_vocab=restrict_decode_vocab, instructions=instructions, labels=labels)
        if 'test' == args.mode:
            evaluator.test()
