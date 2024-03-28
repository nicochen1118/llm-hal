from __future__ import annotations
import sys, os, nltk, json
from typing import Iterable
from tqdm import tqdm
import json
import openai, g4f


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from common.args import args
from evaluate.eval_utils import get_eval_caption


template = """ You don't have to generate explanations, you just have to give me the results. Please act as an impartial and objective judge and calculate the coverage of the Captions for the Objects and Ground Truth, We just need to see if objects of captions cover all objects of the Objects and Ground Truth. you should give me coverage between 1-100 based on the following criteria:

1. captions can use vague term and is no need to mention the specifics likes color or shape or type, include the more objects can get higher coverage score.There is no need to deduct or plus coverage because vague term ,  vague term is inconsequential. captions objects can have a certain amount of ambiguity, covering the type is also considered correct, for example: if captions have fruit, the objects have orange, we consider the fruit cover the orange.
2. Objects in Ground Truth: Compare the objects in the captions with the actual situation, checking if captions covers features consistent with the the ground truth. Only need to find objects in ground truth or Objects is whether in captions.
3. Coverage of Multiple Objects: calculate the coverage of the Captions for the Objects. If the captions contains multiple objects, ensure that the captions covers all these objects rather than focusing on only a subset. There is no need to calculate the coverage of the ground truth for the Objects.


### Captions
{}

### Ground Truth
{}

### Objects
{}
Answer: The coverage is  
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
    if stream:
        for message in response:
            print(message, flush=True, end="")
        print()
    else:
        return response


if __name__ == "__main__":
    image_ids, captions = get_eval_caption()
    with open(os.path.join(args.annotation_path, "captions_train2014.json"), "r") as f:
        data = json.load(f)
    ground_truth = []
    with open(os.path.join(args.annotation_path, "instances_train2014.json"), "r") as f:
        instance = json.load(f)
    instance_truth = []
    category_id = []
    category_name = []

    id2caption = {}
    for image in data["annotations"]:
        id2caption[image["image_id"]] = image["caption"]
    ground_truth = [id2caption[image_id] for image_id in image_ids]

    for image_id in image_ids:
        image_category_id = []
        for image in instance["annotations"]:
            if image["image_id"] == image_id:
                image_category_id.append(image["category_id"])
        category_id.append(image_category_id)
    
    for line in category_id:
        image_category = []
        for ids in line:
            for objects in instance["categories"]:
                if ids == objects["id"]:
                    if objects["name"] not in image_category:
                        image_category.append(objects["name"])
        category_name.append(image_category)

    # ask GPT-4 to evaluate
    scores = []
    for i in range(len(image_ids)):
        input_text = template.format(captions[i], ground_truth[i], category_name[i])
        response = anychat_gpt_4(
            messages=[{"role": "user", "content": input_text}],
        )
        print(response)
        s = ""
        for j in response:
            if j.isalpha() or j == " ":
                continue
            if j == '%' or j == " ":
                break
            s += j
        s = eval(s)
        print(f"Response: {s/100}, {response}")
        scores.append(s/100)
    print("Sum: {}".format(sum(scores)))
    print("Len: {}".format(len(scores)))
    print("Average score: {}".format(sum(scores) / len(scores)))
