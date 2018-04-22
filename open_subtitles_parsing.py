import re
import random
import numpy as np
import pandas as pd

lines = []

with open("data/OpenSubtitles2016.ru.txt", encoding="utf-8") as file:
    for line in file:
        lines.append(line)

total_lines = len(lines)

for i in range(total_lines):
    lines[i] = re.sub(r"(?P<punctuation>[/():;*%',.?!\-\"])", r" \g<punctuation> ", lines[i].lower().strip())
    if i % 1000000 == 0:
        print(i)





generated_dialogues = []

for _ in range(200000):
    i = random.randrange(0, total_lines - 4)
    generated_dialogues.append([0, lines[i], lines[i + 1], lines[i + 2], 0, lines[i + 3], "good", 0.5])

for _ in range(100000):
    i = random.randrange(0, total_lines - 3)
    generated_dialogues.append([0, np.nan, lines[i], lines[i + 1], 0, lines[i + 2], "good", 0.5])

for _ in range(100000):
    i = random.randrange(0, total_lines - 2)
    generated_dialogues.append([0, np.nan, np.nan, lines[i], 0, lines[i + 1], "good", 0.5])


for _ in range(200000):
    i = random.randrange(0, total_lines - 3)
    j = random.randrange(0, total_lines - 1)
    generated_dialogues.append([0, lines[i], lines[i + 1], lines[i + 2], 0, lines[j], "bad", 0.5])

for _ in range(100000):
    i = random.randrange(0, total_lines - 2)
    j = random.randrange(0, total_lines - 1)
    generated_dialogues.append([0, np.nan, lines[i], lines[i + 1], 0, lines[j], "bad", 0.5])

for _ in range(100000):
    i = random.randrange(0, total_lines - 1)
    j = random.randrange(0, total_lines - 1)
    generated_dialogues.append([0, np.nan, np.nan, lines[i], 0, lines[j], "bad", 0.5])


df = pd.DataFrame(generated_dialogues)

df.to_csv("data/bootstrapping_800000.tsv", header=False, sep="\t", index=False)


all_words = set()


def collect_words(s: str):
    if type(s) == str:
        all_words.update(s.split())
    return s


for dialogue in generated_dialogues:
    for x in dialogue:
        collect_words(x)

with open("bootstrapping_800000_words.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(sorted(list(all_words))))
