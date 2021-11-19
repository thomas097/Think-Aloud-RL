from collections import Counter
import re

counter = Counter()

with open('movie_lines.txt', 'r', encoding='ISO-8859-1') as file:
    for line in file.readlines():
        try:
            _, _, _, _, utt = line.strip().split(" +++$+++ ")
        except:
            continue

        tokens = re.findall("[a-z\']+", utt.lower())

        for token in tokens:
            token = token.strip()
            
            if len(token) > 0:
                counter[token] += 1


total = sum(counter.values())
counter = {w:c/total for w, c in counter.items()}

with open('word_probs.txt', 'a', encoding='utf-8') as file:
    for token, prob in counter.items():
        file.write("%s %s\n" % (token, prob))
