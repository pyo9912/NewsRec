import os
import sys
import json
import random
from copy import deepcopy
from tqdm import tqdm

from utils.parser import parse_args
from utils.parser import checkPath

# content_data = json.load((open('./train/rq1.json', 'r', encoding='utf-8')))[0]

def create_test_rq1(args):
    args.rq_num = 1
    input_dir = os.path.join(args.home,"Task1/data/save")
    with open(os.path.join(input_dir,"rq4.json")) as f:
        json_object = json.load(f)
    sample_list = []
    result_list = []
    for i in range(len(json_object)):
        result_list.append(json_object[i])
    
    sample_list = random.sample(result_list, 10000)
    output_dir = os.path.join(args.home,"Task1/data/sample")
    checkPath(output_dir)
    with open(os.path.join(output_dir,"rq4.json"),'w',encoding='utf-8') as result_f:
        result_f.write(json.dumps(sample_list, indent=4))
    result_f.close()
    f.close()




if __name__ == "__main__":
    args = parse_args()
    create_test_rq1(args=args)