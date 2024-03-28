from __future__ import annotations
import sys, os, nltk, json
from typing import Iterable
from tqdm import tqdm
import json
import openai, g4f

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from common.args import args
from evaluate.eval_utils import get_eval_caption_test


template = """Please act as an impartial and objective judge and evaluate the fluency of the captions, 
you can choose the options based on the following criteria:

1. Grammatical Accuracy: Check the sentence's grammatical structure to ensure it adheres to standard grammar rules, avoiding syntactic errors.

2. Smooth Transitions: Evaluate the use of transition words and phrases to ensure logical flow between ideas and smooth transitions from one sentence to the next.

3. Vocabulary Appropriateness: Choose precise and suitable vocabulary to effectively convey the intended meaning and enhance overall clarity.

4. Sentence Structure and Variety: Pay attention vary structures for a more engaging and diverse reading experience. The length of the sentence and the amount of information had no effect on the evaluation

You don't have to generate explanations, you just have to give me the results

### Captions1
{}

### Captions2
{}

Options:
(1) Captions1
(2) Captions2
(3) Both captions are equal

Answer: I will choose Option
"""


openai.api_key = ""
proxy = "http://127.0.0.1:7890"
def anychat_gpt_4(messages: list):
    completion = openai.ChatCompletion.create(model="gpt-4", messages=messages)
    return completion.choices[0].message.content

def g4f_gpt_4(messages: list, stream=True):
    response = g4f.ChatCompletion.create(
        model=g4f.models.gpt_4, provider=g4f.Provider.Bing, messages=messages, stream=stream, proxy=proxy
    )

    return response


if __name__ == "__main__":
    dict1 = get_eval_caption_test("dataset/minigpt_lp1.txt")
    dict2 = get_eval_caption_test("dataset/minigpt_w0.4_lp2.txt")

    same_id = dict1.keys() & dict2.keys()
    dict1 = {key: dict1[key] for key in same_id}
    dict2 = {key: dict2[key] for key in same_id}

    # ask GPT-4 to evaluate
    win_caption1 = 0
    win_caption2 = 0
    equal_caption = 0
    for i, _ in dict1.items():
        input_text = template.format(dict1.get(i), dict2.get(i))
        # print(i)
        # print(dict1.get(i))
        # print(dict2.get(i))
        response = anychat_gpt_4(
            messages=[{"role": "user", "content": input_text}],
        )
        # print("captions1:", captions1[i])
        # print("captions2:", captions2[i])
        print(response)
        for message in response:
            # print(type(message), message)
            if (message == "1"):
                win_caption1 += 1
                break
            if (message == "2"): 
                win_caption2 += 1
                break
            if (message == "3"):
                equal_caption += 1
                break
    print(
        f"Win rate of captions1: {win_caption1 / 500 :.2%}"
        f"\nWin rate of captions2: {win_caption2 / 500:.2%}"
        f"\nRate of equal: {equal_caption / 500:.2%}"
    )


