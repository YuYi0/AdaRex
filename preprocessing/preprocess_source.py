import logging
import sys
import pickle
import gzip
import os
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from itertools import chain
from tqdm import tqdm
import multiprocessing
import pickle
import collections
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')



def process_row(row):
    data = row
    rating = data["overall"]
    review = data["reviewText"]
    summary = data["summary"]
    try: # if summary nan.
        if np.isnan(summary):
            if rating == 1:
                data["summary"] = 'worst'
            elif rating == 2:
                data["summary"]  = 'bad'
            elif rating == 3:
                data["summary"]  = 'average'
            elif rating == 4:
                data["summary"]  = 'good'
            else:
                data["summary"]  = 'best'
    except:
        pass
    
    if summary == "": # if summary ""
        if rating == 1:
            data["summary"] = 'worst'
        elif rating == 2:
            data["summary"]  = 'bad'
        elif rating == 3:
            data["summary"]  = 'average'
        elif rating == 4:
            data["summary"]  = 'good'
        else:
            data["summary"]  = 'best'
    
    try:    # if review nan
        if np.isnan(review):
            data["reviewText"] = data["summary"]
    
    except:
        pass
    
    if review == "":  # if review ""
        data["reviewText"] = data["summary"]

    words = data["reviewText"].split()
    max_length = 256
    if len(words) > max_length: #truncate
        data["reviewText"] = " ".join(words[:max_length])
    return data # a dictionary

def process_all(rows):
    pool = multiprocessing.Pool()
    results = pool.map(process_row, rows)
    pool.close()
    pool.join()
    return results


if __name__ == '__main__':
    
    combo_data = []
    for line in gzip.open("datasets/source/raw/reviews_Electronics_5.json.gz", 'r'):
        review = eval(line)
        combo_data.append(review)
    print(len(combo_data))


    users = []
    user_counter = []

    for datum in combo_data:
        users.append(datum["reviewerID"])

    users = collections.Counter(users)

    for user in users.keys():
        user_counter.append(users[user])

    user_counter = collections.Counter(user_counter) 
    many_users = []
    for key in users.keys():
        if users[key] > 25:
            many_users.append(key)
    df = pd.DataFrame(combo_data)
    df = df[~df["reviewerID"].isin(many_users)]
    df = df.reset_index(drop=True)
    
    
    output_data = process_all(df.iterrows())
    df = pd.DataFrame(output_data)
    df.to_csv("datasets/source/Electronics.csv", index=False, encoding='utf_8_sig')
    print(f"# data: {len(df)}")
    print(f"# user: {len(set(df['reviewerID']))}")
    print(f"# item: {len(set(df['asin']))}")
    words = df["reviewText"].str.split()
    lengths = np.mean([len(item) for item in words])
    print(f"# avg words: {lengths}")


                                