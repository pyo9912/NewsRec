import os
import json
import pandas as pd

## MIND 데이터셋 불러오기
data_path = "../MIND"

MIND_Large = data_path + "/large"
MIND_large_train = MIND_Large + "/train"
MIND_large_valid = MIND_Large + "/dev"
MIND_large_test = MIND_Large + "/test"

MIND_small = data_path + "/small"
MIND_small_train = MIND_small + "/train"
MIND_small_valid = MIND_small + "/dev"
MIND_small_test = MIND_small + "/test"

# news.tsv
# Columns: nid, cat, subcat, title, abstract, URL, Title Entities, Abstract Entites

# news_with_body.tsv
# Columns: nid, cat, subcat, title, abstract, URL, Title Entities, Abstract Entities, Body

news_train = pd.read_csv(os.path.join(MIND_large_train,"news.tsv"),
                         error_bad_lines=False,
                         header=None,
                         sep='\t')

# print(news_train[2].value_counts())
# print(news_train.loc[news_train[1]=='news', [2]].value_counts())

## 딕셔너리 만들기
newsList = []
newsIDs = news_train[0].values.tolist()
newsCat = news_train[1].values.tolist()
newsSubcat = news_train[2].values.tolist()
newsTitles = news_train[3].values.tolist()

for i in range(len(newsIDs)):
    dict = {
        "ID":newsIDs[i],
        "Title":newsTitles[i],
        "Category":newsCat[i],
        "SubCategory":newsSubcat[i]
    }
    # dict = {newsIDs[i]:newsTitles[i]}
    newsList.append(dict)



## json 파일에 쓰기
file_path = "./allNewsData.json"
# file_path = "./newsData.json"

with open(file_path, 'w', encoding='utf-8') as file:
    json.dump(newsList, file, indent="\t")
