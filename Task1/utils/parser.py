import argparse
import json
import logging
import os.path as osp
from typing import Union
import os

import torch
import platform
    

def parse_args():
    parser = argparse.ArgumentParser()
    # common
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--max_input_length', type=int, default=200)
    parser.add_argument('--device_id', type=str, default='0')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        choices=['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-13b-hf', 'meta-llama/Llama-2-7b-chat-hf', 'gpt-3.5-turbo'])
    parser.add_argument('--model_name', type=str, default='llama')
    parser.add_argument('--log_file', type=str, default='')
    parser.add_argument('--chatgpt_cnt', type=int, default=0)
    parser.add_argument('--chatgpt_hit', type=int, default=0)
    parser.add_argument('--chatgpt_key', type=str, default="")
    parser.add_argument('--num_device', type=int, default=1)
    parser.add_argument('--log_name', type=str, default='MYTEST')
    parser.add_argument("--write", action='store_true', help="Whether to write of results.")

    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--home', type=str, default='./')
    parser.add_argument('--time', type=str, default='0000-00-00_000000')
    
    parser.add_argument('--rq_num', type=str, default='1')
    parser.add_argument('--task', type=str, default='1', choices=['1', '2'])

    args = parser.parse_args()

    from datetime import datetime
    from pytz import timezone
    args.time = str(datetime.now(timezone('Asia/Seoul')).strftime('%m%d%H%M%S'))
    

    args.device_id = f'cuda:{args.device_id}' if args.device_id else "cpu"
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
    args.num_device = torch.cuda.device_count()

    args.wandb_project = "LLMCRS"
    args.wandb_run_name = args.time + '_' + args.log_name

    args.output_dir = os.path.join(args.home, 'result')
    if not os.path.exists(args.output_dir): os.mkdir(args.output_dir)

    print(args)
    # logging.info(args)
    args = init_dir(args)
    return args

def checkPath(*args):
    for arg in args:
        if not os.path.exists(arg): os.makedirs(arg)
    
def init_dir(args):
    system_os = platform.platform()
    if "Linux" in system_os:
        args.home = "/content/drive/MyDrive/Colab/NewsRec"
    return args