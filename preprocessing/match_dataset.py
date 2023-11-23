import logging
import sys
import pickle
import gzip

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logging.info("READING DATA...")

# step-1: read .pickle file and raw file. 
file = "reviews_Movies_and_TV_5"

with open("../English-Jar/lei/output/2014_amazon/"+file+".pickle", "rb") as infile:
    explanation_data = pickle.load(infile)
    
raw_data = []
for line in gzip.open("../English-Jar/lei/input/2014_amazon/"+file+".json.gz", 'r'):
    review = eval(line)
    raw_data.append(review)
    
assert len(explanation_data) == len(raw_data)


# step-2: prcess explanation data that keeps only the longest explanation sentence.
logging.info("PROCESSING DATA...")
new_explanation_data = []
for explanation_datum in explanation_data:
    try:
        explanations = explanation_datum["sentence"] # a list of quadtriple.
        max_length = 0
        max_indice = 0
        for num in range(len(explanations)):
            sentence = explanations[num][2]
            length = len(sentence.split())
            if length > max_length:
                max_length = length
                max_indice = num
        explanation_datum["sentence"] = explanation_datum["sentence"][max_indice]  #keep only the longest one
        new_explanation_data.append(explanation_datum)
    except:
        pass
    
    
logging.info("MATCHING DATA...")
# step-3: match explanation and raw data into a larger dictionary.
#"""{'reviewerID': 'AB9S9279OZ3QO',
# 'asin': '0078764343',
# 'reviewerName': 'Alan',
# 'helpful': [1, 1],
# 'reviewText': "I haven't gotten around to playing the campaign but the multiplayer is solid and pretty fun. Includes Zero Dark Thirty pack, an Online Pass, and the all powerful Battlefield 4 Beta access.",
# 'overall': 5.0,
# 'summary': 'Good game and Beta access!!',
# 'unixReviewTime': 1373155200,
# 'reviewTime': '07 7, 2013', 
#.'sentence':(aspect, opinion, explanation, sentiment)}
# """
new_raw_data = []
for explanation_datum in new_explanation_data:
    userID = explanation_datum["user"]
    itemID = explanation_datum["item"]
    for raw_datum in raw_data:
        reviewerID = raw_datum["reviewerID"]
        asin  = raw_datum["asin"]
        if userID == reviewerID and itemID == asin:   # if interaction matched, break the inner loop.
            raw_datum["sentence"] = explanation_datum["sentence"]  # add "sentence" key to current dictionary.
            new_raw_data.append(raw_datum)
            break

logging.info("SAVING DATA...")
save_file = "explanations"+file[7:]+".pickle"
# Open the file in binary mode and save the list
with open(save_file, 'wb') as file:
    pickle.dump(new_raw_data, file)