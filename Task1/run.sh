#!/bin/bash

### LLaMa
## RQ1: ID -> title
# python Task1.py --base_model=meta-llama/Llama-2-7b-chat-hf --batch_size=32 --eval_batch_size=6 --rq_num=1 --mode=train --log_name=rq1 --sample_num=100 
# python Task1.py --base_model=meta-llama/Llama-2-7b-chat-hf --batch_size=32 --eval_batch_size=6 --rq_num=1 --mode=test --test_epoch_num=5 --log_name=rq1 --sample_num=100 --write 
## RQ2: title -> ID
# python Task1.py --base_model=meta-llama/Llama-2-7b-chat-hf --batch_size=32 --eval_batch_size=6 --rq_num=2 --mode=train --log_name=rq2 --sample_num=100 
# python Task1.py --base_model=meta-llama/Llama-2-7b-chat-hf --batch_size=32 --eval_batch_size=6 --rq_num=2 --mode=test --test_epoch_num=1 --sample_num=100 --log_name=rq2 --write
## RQ7: title -> ID
# CUDA_VISIBLE_DEVICES=1 python Task1.py --base_model=meta-llama/Llama-2-7b-chat-hf --batch_size=32 --eval_batch_size=6 --rq_num=7 --mode=train --log_name=rq7 --sample_num=100 
# python Task1.py --base_model=meta-llama/Llama-2-7b-chat-hf --batch_size=32 --eval_batch_size=6 --rq_num=7 --mode=test --test_epoch_num=30 --sample_num=100 --log_name=rq7 --write
## RQ4: title -> cat, sub, id
# python Task1.py --base_model=meta-llama/Llama-2-7b-chat-hf --batch_size=32 --eval_batch_size=6 --rq_num=4 --mode=train --log_name=rq4 --sample_num=100 
# python Task1.py --base_model=meta-llama/Llama-2-7b-chat-hf --batch_size=32 --eval_batch_size=6 --rq_num=4 --mode=test --test_epoch_num=5 --sample_num=100 --log_name=rq4 --write


### T5
## RQ1: ID -> title
# python Task1.py --base_model=google/flan-t5-large --batch_size=16 --eval_batch_size=4 --rq_num=1 --mode=train --log_name=rq1 --sample_num=100
# python Task1.py --base_model=google/flan-t5-large --batch_size=16 --eval_batch_size=4 --rq_num=1 --mode=test --test_epoch_num=1 --log_name=rq1 --sample_num=100 --write
## RQ2: title -> ID
# python Task1.py --base_model=google/flan-t5-large --batch_size=8 --eval_batch_size=4 --rq_num=2 --mode=train --log_name=rq2 --sample_num=100
# python Task1.py --base_model=google/flan-t5-large --batch_size=8 --eval_batch_size=4 --rq_num=2 --mode=test --test_epoch_num=3 --log_name=rq2 --sample_num=100 --write
## RQ4: title -> cat, sub, id
# python Task1.py --base_model=google/flan-t5-large --batch_size=8 --eval_batch_size=4 --rq_num=4 --mode=train --log_name=rq4 --sample_num=100
# python Task1.py --base_model=google/flan-t5-large --batch_size=8 --eval_batch_size=4 --rq_num=4 --mode=test --test_epoch_num=1 --log_name=rq4 --sample_num=100 --write
## RQ7: title -> ID
python Task1.py --base_model=google/flan-t5-large --batch_size=8 --eval_batch_size=4 --rq_num=7 --mode=train --log_name=rq7 --sample_num=100 
# python Task1.py --base_model=google/flan-t5-large --batch_size=8 --eval_batch_size=4 --rq_num=7 --mode=test --test_epoch_num=5 --sample_num=1000 --log_name=rq7 --write


### BERT

## RQ2: title -> ID
# python Task1.py --base_model=bert-base-uncased --batch_size=64 --eval_batch_size=6 --rq_num=4 --log_name=rq4 --category=ID --sample_num=100 --write

## RQ3: title -> category
# python Task1.py --base_model=bert-base-uncased --batch_size=64 --eval_batch_size=6 --rq_num=3 --log_name=rq3 --category=Category --sample_num=100 --write
# python Task1.py --base_model=bert-base-uncased --batch_size=64 --eval_batch_size=6 --rq_num=3 --log_name=rq3 --category=Sub-category --sample_num=100 --write


### DSI

## RQ2: title -> ID 
# CUDA_VISIBLE_DEVICES=1  python Task1.py --base_model=t5-large --device_id=1 --batch_size=32 --eval_batch_size=4 --rq_num=7 --log_name=r7 --mode=train --sample_num=10000  --write
# python Task1.py --base_model=t5-large --device_id=1 --batch_size=32 --eval_batch_size=4 --rq_num=7 --log_name=r7 --mode=test --sample_num=10000 --write