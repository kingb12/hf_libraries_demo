from typing import Dict, Optional

import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5TokenizerFast, T5ForConditionalGeneration

if __name__ == '__main__':
    # load the dataset
    dataset = load_dataset("snli")

    # useful when developing: reduce datasize to just ~1 batch. Then remove when pipeline is ready
    #for split in ('train', 'validation', 'test'):
        #dataset[split] = dataset[split].select(range(128))

    # load T5 model/tokenizer
    tokenizer = T5TokenizerFast.from_pretrained('t5-small')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')

    # Dict for storing translations, of unique EN strings to their FR counterpart. Removes repeated translation
    # relative to directly translating dataset
    unique_strings_en_to_fr: Dict[str, Optional[str]] = {}

    # from initial dataset, get all unique premises & hypotheses and store as keys
    for split in dataset:
        for key in ('premise', 'hypothesis'):
            for unique_str in dataset[split].unique(key):
                # initialize to empty, but with key present
                unique_strings_en_to_fr[unique_str] = None

    # convert to a HF dataset
    all_strings_dataset = Dataset.from_list([{'text': text} for text in unique_strings_en_to_fr])

    # add task prefix with dataset.map (now under key 'prompted')
    task_prefix: str = "translate English to French: "
    all_strings_dataset = all_strings_dataset.map(lambda b: {'prompted': task_prefix + b['text']}, num_proc=32,
                          desc="adding task prefixes")

    # tokenize, cap length at 512 and pad remaining sequences
    tokenized_all_strings_dataset = all_strings_dataset.map(
        lambda batch: tokenizer(batch['prompted'], padding=True, max_length=512),
        batched=True,
        batch_size=1024
    )

    # create dataloader for dataset to translate (use num_workers > 1 and as high of batch size as fits nicely on GPU)
    tokenized_all_strings_dataset.set_format(type="torch", columns=["input_ids"])
    all_strings_dataloader = DataLoader(tokenized_all_strings_dataset, batch_size=512, num_workers=8)

    # put model on GPU and translate all unique strings
    model.eval()
    model.cuda()
    with torch.no_grad():
        results = []
        for batch in tqdm(all_strings_dataloader, desc="converting all strings from en to fr"):
            out = model.generate(input_ids=batch['input_ids'].cuda())
            # batch decode results, make sure to skip special tokens like <s>/</s>, they'll be added again by tokenizer
            # for classification in French
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            results.extend(decoded)

    # iterate over dataset and set dict values
    for i, item in enumerate(all_strings_dataset):
        unique_strings_en_to_fr[item['text']] = results[i]

    # use map to extract
    dataset = dataset.map(lambda item: {'fr_premise': unique_strings_en_to_fr[item['premise']],
                                        'fr_hypothesis': unique_strings_en_to_fr[item['hypothesis']]})

    # push to hub
    dataset.push_to_hub("Brendan/nlp244_french_snli")
