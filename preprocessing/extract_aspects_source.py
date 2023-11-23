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

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def obtain_uiReviews(df):
    uReviews = {}
    iReviews = {}
    for i in range(len(df)):
        uid = df.iloc[i]["reviewerID"]
        iid = df.iloc[i]["asin"]
        review = df.iloc[i]["reviewText"]
        if uid not in uReviews.keys():
            uReviews[uid] = []
        if iid not in iReviews.keys():
            iReviews[iid] = []
        uReviews[uid].append(review)
        iReviews[iid].append(review)
    return uReviews, iReviews


def mining_aspects(entity):
    try:
        reviews = uReviews[entity]
    except:
        reviews = iReviews[entity]
    sentences = []
    for i in range(len(reviews)):
        try:
            sentences.append(sent_tokenize(reviews[i].lower()))
        except:
            sentences.append("nan")
    flatten = list(chain.from_iterable(sentences))
    tagged_words = []
    for sentence in flatten:
        words = word_tokenize(sentence)
        tagged_words.extend(pos_tag(words))

    # Extract nouns from tagged words
    explicit_features = [word for word, pos in tagged_words if pos.startswith('NN')]
    implicit_features = [word for word, pos in tagged_words if pos.startswith('JJ')]

    try:
        explicit_transactions = [[feature] for feature in explicit_features]
        te = TransactionEncoder()
        te_ary = te.fit(explicit_transactions).transform(explicit_transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        # Perform frequent itemset mining
        frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
        frequent_nouns = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == 1)]
        explicit_features = [next(iter(itemset)) for itemset in frequent_nouns.sort_values("support", ascending=False).itemsets]
    except:
        explicit_features = []

    try:
        implicit_transactions = [[feature] for feature in implicit_features]
        te = TransactionEncoder()
        te_ary = te.fit(implicit_transactions).transform(implicit_transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        # Perform frequent itemset mining
        frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)
        frequent_nouns = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: len(x) == 1)]
        implicit_features = [next(iter(itemset)) for itemset in frequent_nouns.sort_values("support", ascending=False).itemsets]
    except:
        implicit_features = []
    result = {entity: (explicit_features, implicit_features)}
    return result


def process_all(entitys):
    pool = multiprocessing.Pool()
    results = pool.map(mining_aspects, entitys)
    pool.close()
    pool.join()
    return results


if __name__ == '__main__':
    path = "./datasets/source"
    for data in ["Movies", "Electronics"]:
        file = os.path.join(path, data)
        df = pd.read_csv(file+".csv")

        logging.info("GROUPING REVIEWS...")
        uReviews, iReviews = obtain_uiReviews(df)

        all_users = uReviews.keys()
        all_items = iReviews.keys()

        logging.info("EXTRACTING USER ASPECTS...")
        user_aspects_list = process_all(all_users)
        user_aspects = {}
        for dictionary in user_aspects_list:
            user_aspects.update(dictionary)

        logging.info("EXTRACTING ITEM ASPECTS...")
        item_aspects_list = process_all(all_items)
        item_aspects = {}
        for dictionary in item_aspects_list:
            item_aspects.update(dictionary)

        logging.info("DONE!")
        with open(file+"/user_aspects.pickle", 'wb') as f:
            pickle.dump(user_aspects, f)
        with open(file+"/item_aspects.pickle", 'wb') as f:
            pickle.dump(item_aspects, f)
            
