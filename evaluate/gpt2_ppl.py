from __future__ import annotations
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import sys, os, nltk, json
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from evaluate.eval_utils import get_ppl_caption
import numpy as np

import numpy as np
import torch
from transformers import BertTokenizer, BertForMaskedLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device)
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def PPL (sentence):
    # Tokenize the input
    # Define special tokens

    # temp = tokenizer.tokenize(sentence, add_special_tokens=True)
    tokens_tensor = tokenizer(sentence, return_tensors="pt")

    tokens_tensor["input_ids"] = torch.cat([torch.tensor([[tokenizer.bos_token_id]]), tokens_tensor["input_ids"]], dim=1)
    tokens_tensor["input_ids"] = torch.cat([tokens_tensor["input_ids"], torch.tensor([[tokenizer.eos_token_id]])], dim=1)
    # print(temp)
    # print(tokenizer.bos_token_id, tokenizer.eos_token_id)
    # print(tokens_tensor)
    # Set up parameters for sliding window
    max_length = model.config.n_positions
    stride = 512
    seq_len = tokens_tensor.input_ids.size(1)

    nlls = []
    prev_end_loc = 0

    # Iterate through the sequence with a sliding window
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc

        # Extract the corresponding window from the sequence
        input_ids = tokens_tensor.input_ids[:, begin_loc:end_loc]
        
        # Clone the input_ids to create target_ids with masked tokens
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # -100 is used to mask tokens

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # Loss is calculated using CrossEntropyLoss
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # Calculate the perplexity
    ppl = torch.exp(torch.stack(nlls).mean())

    return(ppl.item())


if __name__ == "__main__":
    print(1)
    image_ids, captions = get_ppl_caption()
    print(1)
    all_ppl = []
    prefix = "Please describe the image in great detail. Your response should have at least 100 words."
    for i in range(len(captions)):
        print(i)
        ppl = PPL(prefix + captions[i])
        all_ppl.append(ppl)
    all_ppl = np.array(all_ppl)
    
    # print(np.isnan(all_ppl))
    print(
        f"Sum: {sum(all_ppl)}"
        f"\nLen: {len(all_ppl)}"
        f"\nPerplexity: {np.mean(all_ppl[~np.isnan(all_ppl)])}",
    )