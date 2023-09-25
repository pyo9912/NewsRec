# To test whether LLM knows item-feature relationships and item-item relationships
import json
import random
from copy import deepcopy
from tqdm import tqdm

content_data = json.load((open('../newsData.json', 'r', encoding='utf-8')))[0]

title2id_template = ["Following text is a title of the news. '%s'. What is the ID of the news?",
                 "Following text is a title of the news. '%s'. Guess the ID of the news."]
                 
                 
id2title_template = ["Following text is an ID of the news. '%s'. what is the title of the news?",
                 "Following text is an ID of the news. '%s'. Guess the title of the news?"]


### {nid, title} --> rq1 prompt
def create_rq1():
    with open("../newsData.json") as f:
        json_object = json.load(f)
    
    result_list = []
    for i in range(len(json_object)):
        news_dict = json_object[i]
        news_id = next(iter(news_dict.keys()))
        news_title = next(iter(news_dict.values()))
        # 각 type의 template마다 생성
        id2title_tpl = id2title_template[0]%(news_id)
        result_list.append({"Question":id2title_tpl, "Answer":news_title})
        # for tpl in title2id_template:
        #     title2id_tpl = tpl % (news_title)
        #     result_list.append({"Question":title2id_tpl , "Answer":news_id})
        # for tpl in id2title_template:
        #     id2title_tpl = tpl % (news_id)
        #     result_list.append({"Question":id2title_tpl , "Answer":news_title})

    with open("../data/rq1.json",'w',encoding='utf-8') as result_f:
        result_f.write(json.dumps(result_list, indent=4))

def create_sample_rq1():
    with open("../newsData.json") as f:
        json_object = json.load(f)
    
    sample_list = []
    result_list = []
    for i in range(len(json_object)):
        news_dict = json_object[i]
        news_id = next(iter(news_dict.keys()))
        news_title = next(iter(news_dict.values()))
        # 각 type의 template마다 생성
        id2title_tpl = id2title_template[0]%(news_id)
        result_list.append({"Question":id2title_tpl, "Answer":news_title})
    sample_list = random.sample(result_list, 20305)

    with open("../data/sample/rq1.json",'w',encoding='utf-8') as result_f:
        result_f.write(json.dumps(sample_list, indent=4))


if __name__ == "__main__":
    create_rq1()
    create_sample_rq1()
    # create_rq1(id2news)
    # create_rq2(news2id)
