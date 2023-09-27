# To test whether LLM knows item-feature relationships and item-item relationships
import os
import sys
import json
import random
from copy import deepcopy
from tqdm import tqdm

from utils.parser import parse_args
from utils.parser import checkPath

content_data = json.load((open('./newsData.json', 'r', encoding='utf-8')))[0]

title2id_template = ["Following text is a title of the news. '%s'. What is the ID of the news?",
                 "Following text is a title of the news. '%s'. Guess the ID of the news."]
                 
                 
id2title_template = ["Following text is an ID of the news. '%s'. what is the title of the news?",
                 "Following text is an ID of the news. '%s'. Guess the title of the news."]

title2cat_template = ["Following text is a title of the news. '%s'. What is the category of the news?",
                      "Following text is a title of the news. '%s', Guess the category of the news."]


### {nid, title} --> rq1 prompt
def create_rq1(args):
    input_dir = os.path.join(args.home,"Task1")
    with open(os.path.join(input_dir,"newsData.json")) as f:
        json_object = json.load(f)
    
    result_list = []
    for i in range(len(json_object)):
        news_dict = json_object[i]
        news_id = next(iter(news_dict.keys()))
        news_title = next(iter(news_dict.values()))
        # 각 type의 template마다 생성
        title2id_tpl = title2id_template[0]%(news_title)
        result_list.append({"Question":title2id_tpl, "Answer":news_id})
        # for tpl in title2id_template:
        #     title2id_tpl = tpl % (news_title)
        #     result_list.append({"Question":title2id_tpl , "Answer":news_id})
        # for tpl in id2title_template:
        #     id2title_tpl = tpl % (news_id)
        #     result_list.append({"Question":id2title_tpl , "Answer":news_title})

    output_dir = os.path.join(args.home,"Task1/data/train")
    checkPath(output_dir)
    with open(os.path.join(output_dir,f"rq{args.rq_num}.json"),'w',encoding='utf-8') as result_f:
        result_f.write(json.dumps(result_list, indent=4))
    result_f.close()
    f.close()

def create_test_rq1(args):
    input_dir = os.path.join(args.home,"Task1")
    with open(input_dir+"/newsData.json") as f:
        json_object = json.load(f)
    
    sample_list = []
    result_list = []
    for i in range(len(json_object)):
        news_dict = json_object[i]
        news_id = next(iter(news_dict.keys()))
        news_title = next(iter(news_dict.values()))
        # 각 type의 template마다 생성
        # id2title_tpl = id2title_template[1] % (news_id)
        # result_list.append({"Question":id2title_tpl , "Answer":news_title})
        title2id_tpl = title2id_template[1]%(news_title)
        result_list.append({"Question":title2id_tpl, "Answer":news_id})
    sample_list = random.sample(result_list, 20305)

    output_dir = os.path.join(args.home,"Task1/data/test")
    checkPath(output_dir)
    with open(os.path.join(output_dir,f"rq{args.rq_num}.json"),'w',encoding='utf-8') as result_f:
        result_f.write(json.dumps(result_list, indent=4))
    result_f.close()
    f.close()

def create_rq3(args):
    args.rq_num = 3
    input_dir = os.path.join(args.home,"Task1")
    with open(os.path.join(input_dir,"allNewsData.json")) as f:
        json_object = json.load(f)
    
    result_list = []
    for i in range(len(json_object)):
        news_vals = list(json_object[i].values())
        news_id = news_vals[0]
        news_title = news_vals[1]
        news_cat = news_vals[2]
        news_subcat = news_vals[3]
        # 각 type의 template마다 생성
        title2cat_tpl = title2cat_template[0]%(news_title)
        result_list.append({"Question":title2cat_tpl, "Answer":news_cat + "|" + news_subcat})

    output_dir = os.path.join(args.home,"Task1/data/train")
    checkPath(output_dir)
    with open(os.path.join(output_dir,f"rq{args.rq_num}.json"),'w',encoding='utf-8') as result_f:
        result_f.write(json.dumps(result_list, indent=4))
    result_f.close()
    f.close()

def create_test_rq3(args):
    args.rq_num = 3
    input_dir = os.path.join(args.home,"Task1")
    with open(os.path.join(input_dir,"allNewsData.json")) as f:
        json_object = json.load(f)
    
    sample_list = []
    result_list = []
    for i in range(len(json_object)):
        news_vals = list(json_object[i].values())
        news_id = news_vals[0]
        news_title = news_vals[1]
        news_cat = news_vals[2]
        news_subcat = news_vals[3]
        # 각 type의 template마다 생성
        title2cat_tpl = title2cat_template[0]%(news_title)
        result_list.append({"Question":title2cat_tpl, "Answer":news_cat + "|" + news_subcat})
    sample_list = random.sample(result_list, 20305)

    output_dir = os.path.join(args.home,"Task1/data/test")
    checkPath(output_dir)
    with open(os.path.join(output_dir,f"rq{args.rq_num}.json"),'w',encoding='utf-8') as result_f:
        result_f.write(json.dumps(sample_list, indent=4))
    result_f.close()
    f.close()

if __name__ == "__main__":
    args = parse_args()
    # create_rq1(args=args)
    # create_test_rq1(args=args)
    create_rq3(args=args)
    create_test_rq3(args=args)
    # create_rq1(id2news)
    # create_rq2(news2id)
