# To test whether LLM knows item-feature relationships and item-item relationships
import os
import sys
import json
import random
from copy import deepcopy
from tqdm import tqdm

from utils.parser import parse_args
from utils.parser import checkPath

content_data = json.load((open('./allNewsData.json', 'r', encoding='utf-8')))[0]

id2title_template = ["Following text is an ID of the news. <%s>. what is the title of the news?",
                 "Following text is an ID of the news. <%s>. Guess the title of the news.",
                 "Following text is an ID of the news. <%s>. Generate the title of the news.",]
                 
title2id_template = ["Following text is a title of the news. <%s>. What is the ID of the news?",
                 "Following text is a title of the news. <%s>. Guess the ID of the news.",
                 "Following text is a title of the news. <%s>. Generate the ID of the news."]
                 
title2id_template2 = ["The ID of the news <%s> is <%s>. What is the ID of the news <%s>?",
                 "The ID of the news <%s> is <%s>. Guess the ID of the news <%s>.",
                 "The ID of the news <%s> is <%s>. Generate the ID of the news <%s>."]

title2cat_template = ["Following text is a title of the news. <%s>. What is the category and subcategory of the news?",
                      "Following text is a title of the news. <%s>, Guess the category and subcategory of the news.",
                      "Following text is a title of the news. <%s>, Generate the category and subcategory of the news."]

title2all_template = ["Following text is a title of the news. <%s>. What is the category, subcategory and ID of the news?",
                      "Following text is a title of the news. <%s>, Guess the category, subcategory and ID of the news.",
                      "Following text is a title of the news. <%s>, Generate the category, subcategory and ID of the news."]

title2catID_template = ["Following text is a title of the news. <%s>. What is the category ID of the news?",
                      "Following text is a title of the news. <%s>, Guess the category ID of the news.",
                      "Following text is a title of the news. <%s>, Generate the category ID of the news."]

body2catID_template = ["Following text is a body of the news. <%s>. What is the category ID of the news?",
                      "Following text is a body of the news. <%s>, Guess the category ID of the news.",
                      "Following text is a body of the news. <%s>, Generate the category ID of the news."]

### {nid, title} --> rq1 prompt
def create_rq1(args):
    args.rq_num = 1
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
        id2title_tpl = id2title_template[0]%(news_id)
        result_list.append({"Question":id2title_tpl, "Answer":news_title})

    output_dir = os.path.join(args.home,"Task1/data/train")
    checkPath(output_dir)
    with open(os.path.join(output_dir,f"rq{args.rq_num}.json"),'w',encoding='utf-8') as result_f:
        result_f.write(json.dumps(result_list, indent=4))
    result_f.close()
    f.close()

def create_test_rq1(args):
    args.rq_num = 1
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
        id2title_tpl = id2title_template[0]%(news_id)
        result_list.append({"Question":id2title_tpl, "Answer":news_title})
    
    sample_list = random.sample(result_list, 20305)
    output_dir = os.path.join(args.home,"Task1/data/train")
    checkPath(output_dir)
    with open(os.path.join(output_dir,f"rq{args.rq_num}.json"),'w',encoding='utf-8') as result_f:
        result_f.write(json.dumps(sample_list, indent=4))
    result_f.close()
    f.close()

def create_rq2(args):
    args.rq_num = 2
    input_dir = os.path.join(args.home,"Task1")
    with open(os.path.join(input_dir,"allNewsData.json")) as f:
        json_object = json.load(f)

    result_list = []
    for i in range(len(json_object)):
        news_vals = list(json_object[i].values())
        news_id = news_vals[0].split('N')[1]
        news_title = news_vals[1]
        news_cat = news_vals[2]
        news_subcat = news_vals[3]
        # 각 type의 template마다 생성
        title2id_tpl = title2id_template[0]%(news_title)
        result_list.append({"Question":title2id_tpl, "Answer":news_id})

    output_dir = os.path.join(args.home,"Task1/data/train")
    checkPath(output_dir)
    with open(os.path.join(output_dir,f"rq{args.rq_num}.json"),'w',encoding='utf-8') as result_f:
        result_f.write(json.dumps(result_list, indent=4))
    result_f.close()
    f.close()


def create_test_rq2(args):
    args.rq_num = 2
    input_dir = os.path.join(args.home,"Task1")
    with open(os.path.join(input_dir,"allNewsData.json")) as f:
        json_object = json.load(f)

    sample_list = []
    result_list = []
    for i in range(len(json_object)):
        news_vals = list(json_object[i].values())
        news_id = news_vals[0].split('N')[1]
        news_title = news_vals[1]
        news_cat = news_vals[2]
        news_subcat = news_vals[3]
        # 각 type의 template마다 생성
        title2id_tpl = title2id_template[0]%(news_title)
        result_list.append({"Question":title2id_tpl, "Answer":news_id})
    sample_list = random.sample(result_list, 20305)
    output_dir = os.path.join(args.home,"Task1/data/test")
    checkPath(output_dir)
    with open(os.path.join(output_dir,f"rq{args.rq_num}.json"),'w',encoding='utf-8') as result_f:
        result_f.write(json.dumps(sample_list, indent=4))
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
        result_list.append({"Question":title2cat_tpl, "Answer":"The category of the news is <" + news_cat + ">. The subcategory of the news is <" + news_subcat + ">."})

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
        result_list.append({"Question":title2cat_tpl, "Answer":"The category of the news is <" + news_cat + ">. The subcategory of the news is <" + news_subcat + ">."})
    sample_list = random.sample(result_list, 20305)

    output_dir = os.path.join(args.home,"Task1/data/test")
    checkPath(output_dir)
    with open(os.path.join(output_dir,f"rq{args.rq_num}.json"),'w',encoding='utf-8') as result_f:
        result_f.write(json.dumps(sample_list, indent=4))
    result_f.close()
    f.close()


def create_rq4(args):
    args.rq_num = 4
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
        title2all_tpl = title2all_template[0]%(news_title)
        result_list.append({"Question":title2all_tpl, "Answer":"The category of the news is <" + news_cat + ">. The subcategory of the news is <" + news_subcat + ">. The ID of the news is <" + news_id + ">."})

    output_dir = os.path.join(args.home,"Task1/data/train")
    checkPath(output_dir)
    with open(os.path.join(output_dir,f"rq{args.rq_num}.json"),'w',encoding='utf-8') as result_f:
        result_f.write(json.dumps(result_list, indent=4))
    result_f.close()
    f.close()

def create_rq5(args):
    args.rq_num = 5
    input_dir = os.path.join(args.home,"Task1")
    with open(os.path.join(input_dir,"allNewsData.json")) as f:
        json_object = json.load(f)
    
    cat2id = {"sports":"01", "news":"02", "finance":"03", "travel":"04", "lifestyle":"05", "video":"06", "foodanddrink":"07", "weather":"08", "autos":"09", "health":"10", "tv":"11", "music":"12", "entertainment":"13", "movies":"14", "kids":"15", "middleeast":"16", "games":"17", "northamerica":"18"}
    result_list = []
    for i in range(len(json_object)):
        news_vals = list(json_object[i].values())
        news_id = news_vals[0]
        news_title = news_vals[1]
        news_cat = news_vals[2]
        news_subcat = news_vals[3]
        news_cat_id = cat2id[news_cat]
        # 각 type의 template마다 생성
        title2catID_tpl = title2catID_template[0]%(news_title)
        result_list.append({"Question":title2catID_tpl, "Answer":"The category ID of the news is <" + news_cat_id + ">."})

    output_dir = os.path.join(args.home,"Task1/data/train")
    checkPath(output_dir)
    with open(os.path.join(output_dir,f"rq{args.rq_num}.json"),'w',encoding='utf-8') as result_f:
        result_f.write(json.dumps(result_list, indent=4))
    result_f.close()
    f.close()

def create_rq6(args):
    args.rq_num = 6
    input_dir = os.path.join(args.home,"Task1")
    with open(os.path.join(input_dir,"allNewsData.json")) as f:
        json_object = json.load(f)
    
    # cat2id = {"sports":"01", "news":"02", "finance":"03", "travel":"04", "lifestyle":"05", "video":"06", "foodanddrink":"07", "weather":"08", "autos":"09", "health":"10", "tv":"11", "music":"12", "entertainment":"13", "movies":"14", "kids":"15", "middleeast":"16", "games":"17", "northamerica":"18"}
    result_list = []
    for i in range(len(json_object)):
        news_vals = list(json_object[i].values())
        news_id = news_vals[0]
        news_title = news_vals[1]
        news_cat = news_vals[2]
        news_subcat = news_vals[3]
        news_body = news_vals[4]
        # 각 type의 template마다 생성
        body2catID_tpl = body2catID_template[0]%(news_body)
        result_list.append({"Question":body2catID_tpl, "Answer":"The category of the news is <" + news_cat + ">. The subcategory of the news is <" + news_subcat + ">."})

    output_dir = os.path.join(args.home,"Task1/data/train")
    checkPath(output_dir)
    with open(os.path.join(output_dir,f"rq{args.rq_num}.json"),'w',encoding='utf-8') as result_f:
        result_f.write(json.dumps(result_list, indent=4))
    result_f.close()
    f.close()

def create_rq7(args):
    args.rq_num = 7
    input_dir = os.path.join(args.home,"Task1")
    with open(os.path.join(input_dir,"allNewsData.json")) as f:
        json_object = json.load(f)

    result_list = []
    for i in range(len(json_object)):
        news_vals = list(json_object[i].values())
        news_id = news_vals[0].split('N')[1]
        news_title = news_vals[1]
        news_cat = news_vals[2]
        news_subcat = news_vals[3]
        # 각 type의 template마다 생성
        title2id_tpl = title2id_template[0]%(news_title)
        result_list.append({"Question":title2id_tpl, "Answer":int(news_id)})
    output_dir = os.path.join(args.home,"Task1/data/train")
    checkPath(output_dir)
    with open(os.path.join(output_dir,f"rq{args.rq_num}.json"),'w',encoding='utf-8') as result_f:
        result_f.write(json.dumps(result_list, indent=4))
    result_f.close()
    f.close()

def create_rq8(args):
    args.rq_num = 8
    input_dir = os.path.join(args.home,"Task1")
    with open(os.path.join(input_dir,"allNewsData.json")) as f:
        json_object = json.load(f)

    result_list = []
    for i in range(len(json_object)):
        news_vals = list(json_object[i].values())
        news_id = news_vals[0].split('N')[1]
        news_title = news_vals[1]
        news_cat = news_vals[2]
        news_subcat = news_vals[3]
        # 각 type의 template마다 생성
        title2id_tpl = title2id_template2[0]%(news_title, news_id, news_title)
        result_list.append({"Question":title2id_tpl, "Answer":news_id})

    output_dir = os.path.join(args.home,"Task1/data/save")
    checkPath(output_dir)
    with open(os.path.join(output_dir,f"rq{args.rq_num}.json"),'w',encoding='utf-8') as result_f:
        result_f.write(json.dumps(result_list, indent=4))
    result_f.close()
    f.close()


if __name__ == "__main__":
    args = parse_args()
    # create_rq1(args=args)
    # create_test_rq1(args=args)
    # create_rq2(args=args)
    # create_test_rq2(args=args)
    # create_rq3(args=args)
    # create_test_rq3(args=args)
    # create_rq4(args=args)
    # create_rq5(args=args)
    # create_rq7(args=args)
    create_rq8(args=args)