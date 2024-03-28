from __future__ import annotations
import sys, os, nltk, json
from typing import Iterable
from tqdm import tqdm
import json
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from common.args import args
from evaluate.eval_utils import get_caption


if __name__ == "__main__":
    image_ids, captions = get_caption()
    num_right_object = 0
    num_wrong_object = 0
    num_hallucination = 0
    
    for caption in captions:
        for word in caption:
            if word == "{":
                num_right_object += 1
                continue
            if word == "[":
                num_wrong_object += 1
                continue

    for caption in captions:
        for word in caption:
            if word == "[":
                num_hallucination += 1
                break   
    rate_of_hallucination = num_hallucination/50
    rate_of_wrong = num_wrong_object/ (num_right_object + num_wrong_object)
    print(rate_of_wrong, rate_of_hallucination, num_hallucination)

