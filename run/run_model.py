import os
from datasets import load_dataset
import pickle
from utils import *
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
import torch
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
import torch.nn as nn
from torch import optim
import numpy as np
import logging
import sys
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import math
import argparse
from torch.autograd import Function

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class Processor():
    def __init__(self, u_vocab, v_vocab, w_vocab, user_aspects, item_aspects, ifsource):
        self.u = u_vocab
        self.v = v_vocab
        self.w = w_vocab
        self.user_aspects = user_aspects
        self.item_aspects = item_aspects
        self.max_length = 20
        self.ifsource = ifsource

    def __call__(self, sample):
        user = sample["reviewerID"]
        product = sample["asin"]
        uid = self.u[user]  # int type
        pid = self.v[product]  # int type
        rating = sample["overall"]  # float type
        if self.ifsource:
            explanationText = sample["reviewText"].lower()  
            explanation_idx  = self.w(explanationText, padding="max_length", max_length=self.max_length, truncation=True)["input_ids"] # EOS at the end.

        else:    
            explanationText = eval(sample["explanation"])[2].lower()  # str type
            explanation_idx = self.w(explanationText, padding="max_length", max_length=self.max_length, truncation=True)["input_ids"]  # EOS at the end.
        
        user_aspects = " ".join(self.user_aspects[user]).lower()          # str type
        item_aspects = " ".join(self.item_aspects[product]).lower()
    
        user_aspects_idx = self.w(user_aspects, padding="max_length", max_length=64, truncation=True)["input_ids"]  # list of int type
        item_aspects_idx = self.w(item_aspects, padding="max_length", max_length=64, truncation=True)["input_ids"]
        
        final_dict = {"input": torch.tensor(user_aspects_idx+item_aspects_idx+[uid, pid], dtype=torch.long),  # len: 2+50+50.
                      "output": torch.tensor([int(rating)] + explanation_idx, dtype=torch.long)}
        return final_dict


class Regressor(nn.Module):
    def __init__(self, emsize=512):
        super().__init__()
        self.activation = nn.Tanh()
        dropout = 0.2
        self.fc = nn.Sequential(nn.Linear(2 * emsize, emsize),
                                self.activation,
                                nn.Dropout(dropout),
                                nn.Linear(emsize, emsize),
                                self.activation,
                                nn.Dropout(dropout),
                                nn.Linear(emsize, emsize),
                                self.activation,
                                nn.Dropout(dropout),
                                nn.Linear(emsize, emsize),
                                self.activation,
                                nn.Dropout(dropout),
                                nn.Linear(emsize, emsize),
                                self.activation,
                                nn.Dropout(dropout),
                                nn.Linear(emsize, 1))
        self.init_weight()

    def init_weight(self):
        initrange = 0.1
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                layer.weight.data.uniform_(-initrange, initrange)
                layer.bias.data.zero_()

    def forward(self, interaction):
        out_mlp = self.fc(interaction).squeeze()  # shape (N)
        return out_mlp

class D4C(nn.Module):
    def __init__(self, snum_users, snum_items, tnum_users, tnum_items, ntoken, emsize):
        super().__init__()
        self.emsize = emsize
        encoder_layer = nn.TransformerEncoderLayer(d_model=emsize, nhead=2, batch_first=True)
        self.positional_encoding = PositionalEncoding(emsize)
        self.source_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.target_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.source_word_embeddings = nn.Embedding(ntoken, emsize)
        self.target_word_embeddings = nn.Embedding(ntoken, emsize)
        self.source_user_embeddings = nn.Embedding(snum_users, emsize)
        self.source_item_embeddings = nn.Embedding(snum_items, emsize)
        self.target_user_embeddings = nn.Embedding(tnum_users, emsize)
        self.target_item_embeddings = nn.Embedding(tnum_items, emsize)
        self.source_lm_head = nn.Linear(emsize, ntoken, bias=False)
        self.target_lm_head = nn.Linear(emsize, ntoken, bias=False)
        self.source_regressor = Regressor(emsize=emsize)
        self.target_regressor = Regressor(emsize=emsize)
        self.rating_loss_fn = nn.MSELoss()
        self.exp_loss_fn = nn.CrossEntropyLoss(ignore_index=0,label_smoothing=0.05)
        self.discriminator = nn.Sequential(
                                nn.Linear(emsize, 100),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.LayerNorm(100),
                                nn.Linear(100, 100),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.LayerNorm(100),
                                nn.Linear(100, 1))
        self.init_weight()

    def init_weight(self):
        initrange = 0.1
        self.source_user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.source_item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.target_user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.target_item_embeddings.weight.data.uniform_(-initrange, initrange)
        self.source_word_embeddings.load_state_dict(torch.load("./glove.pth"))
        self.target_word_embeddings.load_state_dict(torch.load("./glove.pth"))
        self.source_lm_head.weight = self.source_word_embeddings.weight  # tie
        self.target_lm_head.weight = self.target_word_embeddings.weight  # tie
        
    def forward(self, IDs, aspects, decoder_input_ids, ifsource, alpha):
        device = IDs.device
        batch_size = IDs.shape[0]
        if ifsource:            
            item_embeddings = self.source_item_embeddings(IDs[:, 1]).unsqueeze(1)  # (N,1,512)
            user_embeddings = self.source_user_embeddings(IDs[:, 0]).unsqueeze(1)  # (N, 1, 512)
            aspect_embeddings = self.source_word_embeddings(aspects)
            word_embeddings = self.source_word_embeddings(decoder_input_ids)
            
        else:
            item_embeddings = self.target_item_embeddings(IDs[:, 1]).unsqueeze(1)  # (N, 1,512)
            user_embeddings = self.target_user_embeddings(IDs[:, 0]).unsqueeze(1)  # (N, 1, 512)
            aspect_embeddings = self.target_word_embeddings(aspects)               # (N, 128, 512) 
            word_embeddings = self.target_word_embeddings(decoder_input_ids)       # (N, 20, 512)
            
        src = torch.cat([aspect_embeddings, user_embeddings, item_embeddings, word_embeddings], dim=1)  # (N, 150, 512)
        src = src * math.sqrt(self.emsize)
        src = self.positional_encoding(src)
        tgt_len = decoder_input_ids.shape[1]
        attn_mask = generate_adarex_mask(tgt_len, device=device)
        extended_shape = (batch_size, 130+tgt_len)     # 64 aspects for each entity, plus ID
        key_padding_mask = (aspects==0)
        key_padding_mask = torch.cat((key_padding_mask, torch.zeros((batch_size, 2+tgt_len), device=device, dtype=torch.bool)), dim=1) # shape (N, 150)
        
        if ifsource:
            hidden = self.source_encoder(src=src, mask=attn_mask, src_key_padding_mask=key_padding_mask)  # shape (N, 150, 512)
            interaction = hidden[:,128:130,:].view(-1, self.emsize*2)
            pred_ratings = self.source_regressor(interaction)
            word_dist = self.source_lm_head(hidden[:,130:])          # shape (N, 20, 512)
        else:
            hidden = self.target_encoder(src=src, mask=attn_mask, src_key_padding_mask=key_padding_mask)  # shape (N, 150, 512)
            interaction = hidden[:,128:130,:].view(-1, self.emsize*2)
            pred_ratings = self.target_regressor(interaction)
            word_dist = self.target_lm_head(hidden[:,130:])          # shape (N, 20, 512)
        
        
        user_aspects_hidden_states = hidden[:,:64,:].masked_fill(key_padding_mask[:,:64].unsqueeze(-1), 0.0) 
        user_aspects_hidden_states = user_aspects_hidden_states.sum(dim=1) / (~key_padding_mask[:,:64]).sum(dim=1).unsqueeze(-1)   # (N, 300)
        item_aspects_hidden_states = hidden[:,64:128,:].masked_fill(key_padding_mask[:,64:128].unsqueeze(-1), 0.0) 
        item_aspects_hidden_states = item_aspects_hidden_states.sum(dim=1) / (~key_padding_mask[:,64:128]).sum(dim=1).unsqueeze(-1)  

        mean_aspects_hidden_states = hidden[:,:128,:].masked_fill(key_padding_mask[:,:128].unsqueeze(-1), 0.0)
        mean_aspects_hidden_states = mean_aspects_hidden_states.sum(dim=1) / (~key_padding_mask[:,:128]).sum(dim=1).unsqueeze(-1)  # in shape (N, 300)
        
#         mean_aspects_hidden_states = (user_aspects_hidden_states+item_aspects_hidden_states)/2  # (N, 300)
        aspects_hidden_states = torch.cat([user_aspects_hidden_states, item_aspects_hidden_states, mean_aspects_hidden_states], dim=0) # (3N, 300)
        reverse_feature = ReverseLayerF.apply(aspects_hidden_states, alpha)
        domain_output = self.discriminator(reverse_feature).squeeze()      # in shape (3N)  
        return pred_ratings, word_dist, domain_output
    

    def gather(self, batch, device):
        input, output = batch
        input = input.to(device)  # shape (N, 102)
        output = output.to(device)  # shape (N, 26)
        IDs = input[:,-2:]
        aspects = input[:,:-2]
        ratings = output[:, :1].squeeze().float()
        exps = output[:, 1:]
        decoder_input_ids = T5_shift_right(exps) 
        return IDs, aspects, decoder_input_ids, ratings, exps  # (N, 2), (N, 100), (N, 25), (N), (N, 25)

    def generate(self, IDs, aspects):
        device = IDs.device
        max_len = 20
        batch_size = IDs.shape[0]
        item_embeddings = self.target_item_embeddings(IDs[:, 1]).unsqueeze(1)  # (N,1,512)
        user_embeddings = self.target_user_embeddings(IDs[:, 0]).unsqueeze(1)  # (N, 1, 512)
        aspect_embeddings = self.target_word_embeddings(aspects)               # (N, 100, 512) 
        decoder_input_ids = torch.zeros((batch_size, 1)).fill_(0).long().to(device)  # in shape (N,1)
        for i in range(max_len):
            word_embeddings = self.target_word_embeddings(decoder_input_ids) 
            src = torch.cat([aspect_embeddings, user_embeddings, item_embeddings, word_embeddings], dim=1)  # (N, 127, 512)
            src = src * math.sqrt(self.emsize)
            src = self.positional_encoding(src)
            tgt_len = decoder_input_ids.shape[1]
            attn_mask = generate_adarex_mask(tgt_len,device=device)
            extended_shape = (batch_size, 130+tgt_len)  # 64 aspects for each, plus ID
            key_padding_mask = (aspects==0)
            key_padding_mask = torch.cat((key_padding_mask, torch.zeros((batch_size, 2+tgt_len), device=device, dtype=torch.bool)), dim=1)
            hidden = self.target_encoder(src=src, mask=attn_mask, src_key_padding_mask=key_padding_mask)  
            word_dist = self.target_lm_head(hidden[:,130:])
            output_id = word_dist[:, -1, :].topk(1).indices  # in shape (N, 1)
            decoder_input_ids = torch.cat([decoder_input_ids, output_id], dim=-1)
        return decoder_input_ids[:, 1:]  

    def recommend(self, IDs, aspects):
        device = IDs.device
        batch_size = IDs.shape[0]
        item_embeddings = self.target_item_embeddings(IDs[:, 1]).unsqueeze(1)  # (N,1,512)
        user_embeddings = self.target_user_embeddings(IDs[:, 0]).unsqueeze(1)  # (N, 1, 512)
        aspect_embeddings = self.target_word_embeddings(aspects)               # (N, 100, 512) 
        src = torch.cat([aspect_embeddings, user_embeddings, item_embeddings], dim=1)  # (N, 102, 512)
        src = src * math.sqrt(self.emsize)
        src = self.positional_encoding(src)
        attn_mask = generate_adarex_mask(2, device=device)[:130,:130]     # (130, 130)
        extended_shape = (batch_size, 130)                        
        key_padding_mask = (aspects==0)
        key_padding_mask = torch.cat((key_padding_mask, torch.zeros((batch_size, 2), device=device, dtype=torch.bool)), dim=1)
        hidden = self.target_encoder(src=src, mask=attn_mask, src_key_padding_mask=key_padding_mask)  # shape (N, 127, 512)
        interaction = hidden[:,128:130,:].view(-1, self.emsize*2)
        pred_ratings = self.target_regressor(interaction)
        return pred_ratings


def trainModel(device, sbatch_size, batch_size, source_dataset, train_dataset, valid_dataloader, model, learning_rate, num_epochs, coef1, coef2, log_file_name, save_file):
    torch.manual_seed(43)
    source_dataloader = DataLoader(dataset=source_dataset,
                                   batch_size=sbatch_size,
                                   shuffle=True)
    train_dataloader = DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    rating_loss_fn = nn.MSELoss()
    exp_loss_fn = nn.CrossEntropyLoss(ignore_index=0,label_smoothing=0.05)
    domain_loss_fn = nn.BCEWithLogitsLoss()
    
    endurance = 0
    prev_valid_loss = float('inf')
    len_dataloader = len(train_dataloader)
    for epoch in range(num_epochs):
        model.train()
        avg_loss = 0
        # training
        i = 0
        for sbatch, tbatch in tqdm(zip(source_dataloader, train_dataloader), total=len_dataloader):
            p = float(i + epoch * len_dataloader) / 20 / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            
            # source 
            sIDs, saspects, sdecoder_input_ids, sratings, sexps = model.gather(sbatch, device)
            sbatch_size = sIDs.shape[0]
            device = sIDs.device
            spred, sword_dist, sdomain_output = model(sIDs, saspects, sdecoder_input_ids, True, alpha)
            sloss_r = rating_loss_fn(spred, sratings)
            sloss_e = exp_loss_fn(sword_dist.view(-1, 32128), sexps.reshape(-1))
            sdomain_label = torch.zeros(3*sbatch_size, device=device)
            sdomain_loss = domain_loss_fn(sdomain_output, sdomain_label)
            
            sloss = sloss_r + sloss_e 
            
            # target
            tIDs, taspects, tdecoder_input_ids, tratings, texps = model.gather(tbatch, device)
            tbatch_size = tIDs.shape[0]
            tpred, tword_dist, tdomain_output = model(tIDs, taspects, tdecoder_input_ids, False, alpha)
            tloss_r = rating_loss_fn(tpred, tratings)
            tloss_e = exp_loss_fn(tword_dist.view(-1,32128), texps.reshape(-1))
            tloss = tloss_r + tloss_e
            tdomain_label = torch.ones(3*tbatch_size, device=device)
            tdomain_loss = domain_loss_fn(tdomain_output, tdomain_label)
            
            loss = coef1*sloss + coef2*(sdomain_loss+tdomain_loss) + tloss
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            optimizer.zero_grad()
            avg_loss += loss.item()
        avg_loss = avg_loss / len(train_dataloader)
        # logging
        log_file = open(log_file_name, "a")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the current time
        log_file.write(f"Epoch {epoch + 1}: [{current_time}] [lr: {learning_rate}] Loss = {avg_loss:.4f}\n")
        log_file.close()

        # checking learning rate
        current_valid_loss = validModel(model, valid_dataloader, device)
        if current_valid_loss > prev_valid_loss:
            learning_rate /= 2.0
            endurance += 1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        prev_valid_loss = current_valid_loss

        # save
        torch.save(model.state_dict(), save_file)
        if endurance >= 10:
            break

def validModel(model, valid_dataloader, device):
    model.eval()
    with torch.no_grad():
        avg_loss = 0
        for batch in valid_dataloader:
            IDs, aspects, decoder_input_ids, ratings, exps = model.gather(batch, device)
            pred, word_dist, _ = model(IDs, aspects, decoder_input_ids, False, 1)
            loss_r = model.rating_loss_fn(pred, ratings)
            loss_e = model.exp_loss_fn(word_dist.view(-1, 32128), exps.reshape(-1))
            loss = loss_r + loss_e
            avg_loss += loss.item()
    return avg_loss / len(valid_dataloader)


def evalModel(model, test_dataloader, device):
    model = model.to(device)
    model.eval()
    prediction_ratings = []
    ground_truth_ratings = []
    prediction_exps = []
    reference_exps = []
    with torch.no_grad():
        for batch in test_dataloader:
            IDs, aspects, decoder_input_ids, ratings, exps = model.gather(batch, device)
            pred_ratings = model.recommend(IDs, aspects)
            pred_exps = model.generate(IDs, aspects)
            prediction_ratings.extend(pred_ratings.tolist())
            ground_truth_ratings.extend(ratings.tolist())
            prediction_exps.extend(w_vocab.batch_decode(pred_exps, skip_special_tokens=True))
            reference_exps.extend(w_vocab.batch_decode(exps, skip_special_tokens=True))

    prediction_ratings = np.array(prediction_ratings)
    ground_truth_ratings = np.array(ground_truth_ratings)
    rating_diffs = prediction_ratings - ground_truth_ratings
    mae = round(np.mean(np.abs(rating_diffs)), 4)
    rmse = round(np.sqrt(np.mean(np.square(rating_diffs))), 4)
    text_results = evaluate_text(prediction_exps, reference_exps)
    return {"recommendation": {"mae": mae, "rmse": rmse}, "explanation": text_results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="Device index to run the model on")
    parser.add_argument("--log", type=str, default="log.txt", help="Path to the log file")
    parser.add_argument("--save", type=str, default="model_state_dict.pth", help="Path to save the model state dict")

    args = parser.parse_args()
    device = args.device
    log_file_name = args.log
    save_file = args.save
    
    torch.manual_seed(42)
    logging.info("RESOURCE PREPARING ...")
    spath = "../../dataset/Source/Electronics/"
    tpath = "../../dataset/CDs/"  # "Instrument", "Games", "CDs".
    ttrain = os.path.join(tpath, "train.csv")
    tvalid = os.path.join(tpath, "valid.csv")
    ttest = os.path.join(tpath, "test.csv")
    sdataset = load_dataset("csv", data_files=spath+"data.csv")
    tdatasets = load_dataset("csv", data_files={"train": ttrain,
                                                "valid": tvalid,
                                                "test": ttest})
    with open(spath+"/user_aspects.pkl", "rb") as f:
        suser_aspects = pickle.load(f)
    with open(spath+"/item_aspects.pkl", "rb") as f:
        sitem_aspects = pickle.load(f)
    with open(tpath+"/user_aspects.pkl", "rb") as f:
        tuser_aspects = pickle.load(f)
    with open(tpath+"/item_aspects.pkl", "rb") as f:
        titem_aspects = pickle.load(f)

    w_vocab = T5Tokenizer.from_pretrained("t5-small")
    su_vocab = UIvocab(sdataset["train"]["reviewerID"])
    sv_vocab = UIvocab(sdataset["train"]["asin"])
    ifsource = True
    sprocessor = Processor(su_vocab, sv_vocab, w_vocab, suser_aspects, sitem_aspects, ifsource)
    sencoded_data = sdataset.map(lambda sample: sprocessor(sample), num_proc=20)
    sencoded_data.set_format("torch")
    tu_vocab = UIvocab(tdatasets["train"]["reviewerID"])
    tv_vocab = UIvocab(tdatasets["train"]["asin"])   
    ifsource = False
    tprocessor = Processor(tu_vocab, tv_vocab, w_vocab, tuser_aspects, titem_aspects, ifsource)
    tencoded_data = tdatasets.map(lambda sample: tprocessor(sample), num_proc=20)
    tencoded_data.set_format("torch")
    batch_size = 50                
    sbatch_size = int(len(sencoded_data["train"])/len(tencoded_data["train"])) * batch_size
    source_dataset = TensorDataset(sencoded_data["train"]["input"], sencoded_data["train"]["output"])
    train_dataset = TensorDataset(tencoded_data["train"]["input"], tencoded_data["train"]["output"])
    valid_dataset = TensorDataset(tencoded_data["valid"]["input"], tencoded_data["valid"]["output"])
    test_dataset = TensorDataset(tencoded_data["test"]["input"], tencoded_data["test"]["output"])
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    snum_users = len(su_vocab)
    snum_items = len(sv_vocab)
    tnum_users = len(tu_vocab)
    tnum_items = len(tv_vocab)
    ntoken = 32128
    emsize = 300
    logging.info("MODEL LEARNING ...")
    learning_rate = 1e-3
    coef1 = 1
    coef2 = 1
    log_file = open(log_file_name, "a")
    log_file.write(f"coef1: {coef1}, coef2: {coef2}\n")
    log_file.close()
    model = D4C(snum_users, snum_items, tnum_users, tnum_items, ntoken, emsize)
    num_param = count_param(model)
    logging.info(f"trainable params: {num_param}")
    num_epochs = 3
    trainModel(device, sbatch_size, batch_size, source_dataset, train_dataset, valid_dataloader, model, learning_rate, num_epochs, coef1, coef2, log_file_name, save_file)
    logging.info("MODEL EVALUATING ...")
    model.load_state_dict(torch.load(save_file))
    final = evalModel(model, test_dataloader, device)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the current time
    log_file = open(log_file_name, "a")
    log_file.write(
        "------------------------------------------FINAL RESULTS------------------------------------------\n")
    log_file.write(f"[{current_time}] \n")
    log_file.write(
        f"[Recommendation] MAE = {final['recommendation']['mae']} | RMSE = {final['recommendation']['rmse']} \n")
    log_file.write(
        f"[Explanation] ROUGE: {final['explanation']['rouge']['1']}, {final['explanation']['rouge']['2']}, {final['explanation']['rouge']['l']} \n")
    log_file.write(
        f"[Explanation] BLEU: {final['explanation']['bleu']['1']}, {final['explanation']['bleu']['2']}, {final['explanation']['bleu']['3']}, {final['explanation']['bleu']['4']} \n")
    log_file.write(
        f"[Explanation] DIST: {final['explanation']['dist']['1']}, {final['explanation']['dist']['2']}, {final['explanation']['dist']['l']},\n")
    log_file.write(f"[Explanation] METEOR: {final['explanation']['meteor']} \n")
    log_file.write(f"[Explanation] BERT: {final['explanation']['bert']} \n")
    log_file.close()
    logging.info("DONE.")