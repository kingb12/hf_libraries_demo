from collections import Counter
from typing import List, Dict, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datasets import DatasetDict, load_dataset

from hf_libraries_demo.datasets_examples.pre_process_example import pre_process_dataset

if __name__ == '__main__':
    # loading the dataset will download if not present in your cache, or simply load from there
    dataset: DatasetDict = load_dataset("SetFit/20_newsgroups")
    dataset = pre_process_dataset(dataset)

    # Utilizing `unique` to see distinct classes! We don't actually re-use this, but for reference, this
    # function is extremely fast for any column in a HF dataset
    labels = dataset['train'].unique('label_text')

    # count occurrences of each label, package into a dataframe, and then plot
    label_counts: Counter[str] = Counter()
    for item in dataset['train']:
        label_counts[item['label_text']] += 1
    data: List[Dict[str, Union[int, str]]] = [
        {'label': label, 'count': count} for label, count in label_counts.most_common()
    ]

    df: pd.DataFrame = pd.DataFrame(data)
    # orient to h so that labels print nicely. notice swapped (x, y)
    sns.barplot(data=df, x='count', y='label', orient="h")
    plt.tight_layout()
    plt.savefig("plots/label_counts.png")

