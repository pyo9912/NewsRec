import os
import sys
import json
import random
from copy import deepcopy
from tqdm import tqdm


# content_data = json.load((open('./train/rq1.json', 'r', encoding='utf-8')))[0]

def create_test_rq1():
    with open("./rq7.json") as f:
        json_object = json.load(f)
    
    news_without_dup = list({obj["Question"]: obj for obj in json_object}.values())

    output_dir = ("./")
    output_file = ("rq17.json")
    with open(os.path.join(output_dir,output_file),'w',encoding='utf-8') as result_f:
        result_f.write(json.dumps(news_without_dup, indent=4))

    result_f.close()
    f.close()




if __name__ == "__main__":
    create_test_rq1()