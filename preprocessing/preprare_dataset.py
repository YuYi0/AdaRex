import sys
# add the path
sys.path.append("../../")
from base_utils import *
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from functools import partial
from transformers import AutoTokenizer, AutoModel, BartForConditionalGeneration
import argparse
from nltk.tokenize import sent_tokenize
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoTokenizer, BertForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
base = BertForSequenceClassification.from_pretrained("bert-base-uncased")

def statistics(df):
    """
    Do following statistics: 
        # users
        # items
        # interactions
        Avg. c
    """
    users = df["users"]
    items = df["items"]
    u_vocab = UIvocab(users)
    v_vocab = UIvocab(items)
    inter = len(df)
    return {"nusers": len(u_vocab),
            "nitems": len(v_vocab),
            "ninter": len(df)}

def filter_few(df, least_times=5):
    """
    Filter user/item with less than 10 interactions. 
    """
    user_set = list(set(df["users"]))
    item_set = list(set(df["items"]))
    user_interaction = {}
    item_interaction = {}
    # obtain a dictionary that contains the number of interactions for each user/item.
    for i in range(len(df)):
        user = df["users"][i]
        item = df["items"][i]
        if user not in user_interaction.keys():
            user_interaction[user] = 1
        else: 
            user_interaction[user] += 1
            
        if item not in item_interaction.keys():
            item_interaction[item] = 1
        else:
            item_interaction[item] += 1
            
    # record the index of row either the user or the item's total interaction is less than 10. 
    user_row_index = []
    item_row_index = []
    for i in range(len(df)):
        user = df["users"][i]
        item = df["items"][i]
        if user_interaction[user] < least_times:
            user_row_index.append(i)
        if item_interaction[item] < least_times:
            item_row_index.append(i)
    drop_rows = list(set(user_row_index+item_row_index))  # overlap

    # then drop by the row_index
    df = df.drop(df.index[drop_rows]).reset_index(drop=True)
    return df

def obtain_edu(df):
    # conver the string text to a list of edu.    
    all_explanations = []
    drop_rows = []
    for i in tqdm(range(len(df)),leave=False):
        try:
            review = df["explanations"][i]
            explanations = sent_tokenize(review)
            it_copy = copy.deepcopy(explanations)
            for item in it_copy:
                tokens = tokenizer(item, add_special_tokens=False)["input_ids"]
                if len(tokens) < 5 or len(tokens) > 25:
    #             if len(tokenize(item)) < 5 or len(tokenize(item)) > 25:
                    explanations.remove(item)
            all_explanations.append(explanations)
            if len(explanations) < 1:    # for null explanation data, remove it. 
                drop_rows.append(i)
        except:
            all_explanations.append(" ")
            drop_rows.append(i)
            continue
    df["explanations"] = all_explanations
    df = df.drop(df.index[drop_rows]).reset_index(drop=True)   # if there is null explanation, drop it. 
    return df

# Using trained classifier model to recognize the qualified". 
class Classifier(JoModule):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, input_ids):
        logits = self.base(input_ids).logits
        return logits   # in shape (N, 2)
    
    def training_step(self, batch, device):
        input_ids, target = batch
        input_ids = input_ids.to(device)
        target = target.reshape(-1).to(device)
        logits = self.forward(input_ids)
        loss = self.loss_fn(logits, target)
        return loss
        
    def validation_step(self, batch, device, metrics=["loss", "precision"]):
        input_ids, target = batch
        input_ids = input_ids.to(device)
        target = target.reshape(-1).to(device)
        logits = self.forward(input_ids)
        # compute loss
        loss = self.loss_fn(logits, target)
        # compute precision
        precision = (logits.topk(1).indices.reshape(-1) == target).sum() / len(target)
        return {"loss": loss, "precision": precision}

def obtain_offsets(alist): 
    offsets = [0]
    for i,item in enumerate(alist):
        offsets.append(offsets[i]+item)
    return offsets

if __name__ == "__main__":
    # ["Books.json", "Movies_and_TV.json", "Electronics.json", "CDs_and_Vinyl.json", "Kindle_Store.json", 
    # "Home_and_Kitchen.json", "Apps_for_Android.json", "Video_Games.json", "Health_and_Personal_Care.json", "Sports_and_Outdoors.json"]
    names = ["arts", "automotive", "cell", "grocery", "musical", "pet", "sports", "tools"]
    for name in names:
        path = "./"+name+".json"
        data = load_json(path)
        df = pd.DataFrame(data)
        df = df[["reviewerID", "asin", "overall", "reviewText"]]
        df.columns = ["users", "items", "ratings", "explanations"]
        print("start EDU processing...")
        df = obtain_edu(df)
        print("start classifying...")
        model = Classifier(base)
        model = load_model(model, "./saved/exp_classifier_new_dict")
        model = model.to("cuda:1")

        lengths = [len(df["explanations"][i]) for i in range(len(df))]
        offsets = obtain_offsets(lengths)
        explanations_flatten = list(flatten(df["explanations"]))
        predictions = []

        batch_size=500
        nums = int(len(explanations_flatten)/batch_size) 
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(nums), leave=False):
                batch = explanations_flatten[i*batch_size: (i+1)*batch_size]
                input_ids = tokenizer(batch, padding=True, return_tensors="pt").input_ids.to("cuda:1")
                pred = model(input_ids).topk(1).indices.reshape(-1).tolist()
                predictions.extend(pred)
        print(f"explanation ratio: {np.sum(predictions)/len(predictions)}")

        # rearrange the data to table form. 
        drop_rows = []
        new_explanations = []
        for i in range(len(offsets)-1):   ### retrieve.
            # one cell is one row of the dataframe
            cell_explanations = explanations_flatten[offsets[i]:offsets[i+1]]
            cell_labels = predictions[offsets[i]:offsets[i+1]]
            if np.sum(cell_labels) < 1:  # if no qualified explanation, mark it and then drop.
                drop_rows.append(i)
            elif np.sum(cell_labels) == 1:
                idx = np.argmax(cell_labels)
                new_explanations.append(cell_explanations[idx])
            else:  # if more than 2 are labeled as explanation, we select the longest one. 
                qualified_idx =  (np.array(cell_labels) == 1).nonzero()[0]    
                lengths = []
                for idx in qualified_idx:
                    lengths.append(len(cell_explanations[idx]))
                longest_idx = qualified_idx[np.argmax(qualified_idx)]
                new_explanations.append(cell_explanations[longest_idx]) 
        df = df.drop(df.index[drop_rows]).reset_index(drop=True)
        df["explanations"] = new_explanations

        print("post-processing...")
        for i in tqdm(range(50), leave=False):
            df = filter_few(df, 5)
        
        # lowercase
        lower_sentences = []
        for i in tqdm(range(len(df))):
            sentence = df.explanations[i].lower()
            lower_sentences.append(sentence)
        df["explanations"] = lower_sentences
        print(f"final resulting records num:{len(df)}")
        write_csv(df, name+".csv")