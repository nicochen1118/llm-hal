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
    lines = []
    for ids, caption in zip(image_ids, captions): 
        result = ids+"### 0 ### "
        matches_right = re.findall(r'\{([^}]*)\}', caption)
        for i in matches_right:
            result += i
            result += ", "
        matches_wrong = re.findall(r'\[([^]]*)\]', caption)
        for i in matches_wrong:
            result += "["
            result += i
            result += "]"
            result += ", "
        if result.endswith(', '):
            result = result[:-2]
        lines.append(result)
    print(lines)
    with open('dataset/temp.txt', 'w') as file:
    # 将每行文本写入文件
        for line in lines:
            file.write(line + '\n')