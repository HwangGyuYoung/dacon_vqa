from collections import Counter
import json
import os

import pandas as pd

from dataset import base_path

train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
counter = Counter(train_df['answer'])
sorted_dict = sorted(counter.items(), key=lambda item: item[1], reverse=True)

with open('answer2label.txt', mode="w", encoding="utf-8") as writer:
    for i, (k, _) in enumerate(sorted_dict[:3129]):
        to_json = {
            "answer": k,
            "label": i
        }
        writer.write("%s\n" % json.dumps(to_json))
