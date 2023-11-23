import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
from rouge import rouge
from nltk import word_tokenize
import evaluate
from bleu import compute_bleu
import numpy as np
import torch 
import math

class UIvocab():
    def __init__(self, ID_list, specials=None):
        self.ID2idx = {}
        self.idx2ID = {}
        if specials is not None:
            if isinstance(specials, str):
                self.ID2idx[specials] = 0
                self.idx2ID[0] = specials
            if isinstance(specials, list):
                for i, item in enumerate(specials):
                    self.ID2idx[item] = i
                    self.idx2ID[i] = item
        for ID in ID_list:
            # Allows only for string type.
            if isinstance(ID, str):
                pass
            else:
                ID = str(ID)
            if ID not in self.ID2idx.keys():
                length = len(self.ID2idx)
                self.ID2idx[ID] = length
                self.idx2ID[length] = ID

    def lookup_tokens(self, idx_list):
        IDs = []
        for idx in idx_list:
            IDs.append(self.idx2ID[idx])
        return IDs

    def lookup_token(self, idx):
        return self.idx2ID[idx]

    def __getitem__(self, IDs):
        try:
            return self.ID2idx[IDs]
        except:
            # out of Vocab
            return self.ID2idx["<unk>"]

    def __len__(self):
        return len(self.ID2idx)

    
def unique_sentence_percent(sequence_batch):
    def two_seq_same(sa, sb):
        if len(sa) != len(sb):
            return False
        for (wa, wb) in zip(sa, sb):
            if wa != wb:
                return False
        return True    
    unique_seq = []
    for seq in sequence_batch:
        count = 0
        for uni_seq in unique_seq:
            if two_seq_same(seq, uni_seq):
                count += 1
                break
        if count == 0:
            unique_seq.append(seq)

    return round(len(unique_seq) / len(sequence_batch)*100, 2)
    
    
def xavier_normal_initialization(module):
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)

            
            
def evaluate_text(predictions, references):
    """
    Example:
        >>> predictions = ["good day", "need to work"]
        >>> references = ["nice day", "work from home"]
        >>> evlauate_text(predictions, references)
    """
    # compute bleu
    # compute rouge
    # compute distinct
    # compute meteor
    
    def distinct_score(sentences, n):
        sentences = [word_tokenize(sentence) for sentence in sentences]
        unique_ngrams = set()
        total_ngrams = 0

        for sentence in sentences:
            ngrams = [tuple(sentence[i:i + n]) for i in range(len(sentence) - n + 1)]
            unique_ngrams.update(ngrams)
            total_ngrams += len(ngrams)

        distinct_score = len(unique_ngrams) / total_ngrams
        return distinct_score
    # dist score
    try:
        dist1 = round(distinct_score(predictions, 1) * 100, 2)
    except:
        dist1 = 0
    try:
        dist2 = round(distinct_score(predictions, 2) * 100, 2)
    except:
        dist2 = 0
    
    # bleu score
    predictions_tokens = [word_tokenize(prediction) for prediction in predictions]
    references_tokens = [word_tokenize(reference) for reference in references]
    formatted_ref = [[ref] for ref in references_tokens]
    try:
        bleu1, _, _, _, _, _ = compute_bleu(formatted_ref, predictions_tokens, max_order=1, smooth=False)
        bleu1 = round(bleu1*100, 2)
    except:
        bleu1 = 0
    try:
        bleu2, _, _, _, _, _ = compute_bleu(formatted_ref, predictions_tokens, max_order=2, smooth=False)
        bleu2 = round(bleu2*100, 2)
    except:
        bleu2 = 0
    try:
        bleu3, _, _, _, _, _ = compute_bleu(formatted_ref, predictions_tokens, max_order=3, smooth=False)
        bleu3 = round(bleu3*100, 2)
    except:
        bleu3 = 0
    try:
        bleu4, _, _, _, _, _ = compute_bleu(formatted_ref, predictions_tokens, max_order=4, smooth=False)
        bleu4 = round(bleu4*100,2)
    except:
        bleu4 = 0
    
    # rouge score
    score = rouge(predictions, references)
    rouge_s = {k: round(v * 100, 2) for (k, v) in score.items()}

    # meteor score
    meteor = evaluate.load('meteor')
    try:
        meteor_score = meteor.compute(predictions=predictions, references=references)["meteor"]
        meteor_score = round(meteor_score*100, 2)    
    except:
        meteor_score = 0
    
  
    # bert_score
    bertscore = evaluate.load("bertscore")
    bert_score = bertscore.compute(predictions=predictions, references=references, model_type="bert-base-uncased", lang="en")
    bert_score = round(np.mean(bert_score["f1"])*100,2)
    
    # USR
    USR = unique_sentence_percent(predictions)
    
    return {
            "rouge": {"1":rouge_s["rouge_1/f_score"], "2":rouge_s["rouge_2/f_score"], "l":rouge_s["rouge_l/f_score"]},
            "bleu": {"1":bleu1, "2":bleu2, "3":bleu3, "4":bleu4}, 
            "dist": {"1":dist1, "2":dist2, "l": USR},
            "meteor": meteor_score, 
            "bert":bert_score}


def T5_shift_right(input_ids):
    decoder_start_token_id = 0
    pad_token_id = 0

    assert decoder_start_token_id is not None, (
        "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id."
        " See T5 docs for more information"
    )
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def create_mask(source, target):
    device = source.device
    PAD_IDX = 0
    src_seq_len = source.shape[1]
    tgt_seq_len = target.shape[1]

    # attention mask
    src_mask = torch.zeros((src_seq_len, src_seq_len), dtype=torch.bool, device=device)
    tgt_mask = generate_square_mask(tgt_seq_len)
    tgt_mask = tgt_mask.to(device)

    # padding mask
    src_padding_mask = (source == PAD_IDX).to(device)
    tgt_padding_mask = (target == PAD_IDX).to(device)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def generate_square_mask(seqlen, device):
    mask = torch.triu(torch.ones((seqlen, seqlen), device=device), diagonal=1) == 1
    return mask

def generate_peter_mask(tgt_len, device):
    src_len = 2
    total_len = src_len + tgt_len
    mask = generate_square_mask(total_len, device)
    mask[0, 1] = False  # allow to attend for user and item
    return mask

def generate_adarex_mask(tgt_len, device):
    # cross interaction.
    num_aspect = 64
    IDs = 2
    tgt_len= tgt_len
    total_len = num_aspect*2 + IDs + tgt_len
    # creat mask 1: aspect mask. upper lef.
    mask1 = torch.full((num_aspect*2, num_aspect*2), True, dtype=torch.bool, device = device)
    mask1[:num_aspect, :num_aspect] = False
    mask1[num_aspect:, num_aspect:] = False
    # create mask2: right zeros. upper right.
    mask2 = torch.ones((num_aspect*2, IDs+tgt_len), dtype=torch.bool, device = device)
    mask2 = torch.cat((mask1, mask2), dim=1)
    # create mask3: lower left.
    mask3 = torch.ones((2 + tgt_len, 2 * num_aspect), dtype=torch.bool, device = device)
    mask3[0, :num_aspect] = False
    mask3[1, num_aspect:] = False
    # create mask4: explanation.   lower right. 
    mask4 = generate_peter_mask(tgt_len, device)
    mask4 = torch.cat([mask3, mask4], dim=1)
    mask = torch.cat([mask2, mask4], dim=0)
    return mask


def count_param(model):
    from numerize import numerize
    count = 0
    for parameter in model.parameters():
        if parameter.requires_grad:
            count += len(parameter.view(-1))
    return numerize.numerize(count)