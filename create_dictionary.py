import numpy as np
import pandas as pd

train_data = pd.read_csv("data/train.tsv", names=["context_id", "context_2", "context_1", "context_0",
                                                  "reply_id", "reply", "label", "confidence"], header=None, sep="\t",
                         quoting=3)

public_data = pd.read_csv("data/public.tsv", names=["context_id", "context_2", "context_1", "context_0",
                                                    "reply_id", "reply"], header=None, sep="\t",
                          quoting=3)

final_data = pd.read_csv("data/final.tsv", names=["context_id", "context_2", "context_1", "context_0",
                                                    "reply_id", "reply"], header=None, sep="\t",
                          quoting=3)

all_words = set()


def collect_words(s: str):
    if type(s) == str:
        all_words.update(s.split())
    return s


train_data[["context_2", "context_1", "context_0", "reply"]].applymap(collect_words)
public_data[["context_2", "context_1", "context_0", "reply"]].applymap(collect_words)
final_data[["context_2", "context_1", "context_0", "reply"]].applymap(collect_words)

with open("train_and_public_and_final_words.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(sorted(list(all_words))))
