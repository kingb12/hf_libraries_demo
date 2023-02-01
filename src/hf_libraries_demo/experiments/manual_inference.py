from pprint import pprint

import torch
from datasets import DatasetDict, load_dataset
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, BertTokenizer

from hf_libraries_demo.datasets.pre_process_example import pre_process_dataset

if __name__ == '__main__':
    # load and pre-process our dataset
    dataset: DatasetDict = load_dataset("SetFit/20_newsgroups")
    dataset = pre_process_dataset(dataset)

    # instantiate tokenizer, model, and config
    config: AutoConfig = AutoConfig.from_pretrained("microsoft/xtremedistil-l6-h384-uncased")

    # happens to use the BERT tokenizer! don't guess!
    tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model: AutoModelForSequenceClassification = \
        AutoModelForSequenceClassification.from_pretrained("microsoft/xtremedistil-l6-h384-uncased",
                                                           # we get a new classification head! Need number of labels
                                                           num_labels=20)

    # quick test: manually running one example
    datapoint = dataset['train'][0]
    model_input = tokenizer(datapoint['text'], return_tensors="pt")  # return_tensors="pt" gives us a PyTorch tensor
    # we should get back a long tensor for input ids!
    assert type(model_input['input_ids']) == torch.Tensor
    assert model_input['input_ids'].dtype == torch.int64
    assert 'attention_mask' in model_input  # note: our attention mask is all ones, since we didn't mask anything
    assert model_input['token_type_ids'].sum() == 0  # all token types are real (non-mask) tokens
    model_output = model(**model_input)
    pprint(model_output)

    # we get logits (log probabilities per class) from the model. Need to compute a prediction ourselves!
    logits: Tensor = model_output['logits']
    # highest log prob -> highest prob. squeeze() because we're ignoring batching
    prediction: int = torch.argmax(logits.squeeze()).item()

    # aside: we can recover input text from a tokenizer using decode. Useful for prediction in gen tasks AND checking
    # current input states. Again, squeeze to remove batch=1
    input_text: str = tokenizer.decode(model_input['input_ids'].squeeze())
    print(f"BERT input: {input_text}")  # Additional [CLS], [SEP] tokens come from BERT tokenizer config!
    print(f"Predicted: {prediction}")

    # Inference on all examples: manual again

    # select selects only training set examples corresponding to indices in argument (e.g. the first 100)
    small_train_set = dataset['train'].select(range(100))

    # datasets cooperate very nicely with tokenizers! this will take the contents of text, and
    # add attributes for input_ids, attention_mask, etc, truncating sentences at 510 tokens (from tokenizer config!)
    tokenized_train_set = small_train_set.map(lambda item: tokenizer(item['text'], truncation=True))
    assert len(tokenized_train_set) == 100
    assert max(len(i['input_ids']) for i in tokenized_train_set) == 512  # our sequences should be of correct length!

    # we can also tokenize in a batch trivially:
    tokenized_train_set = small_train_set.map(lambda batch: tokenizer(batch['text'], truncation=True), batched=True)

    print(tokenized_train_set[0])
    # we can also tensorize things with the set format function (this turns 'input_ids', 'label', etc. into tensors)
    tokenized_train_set.set_format("pt", columns=["input_ids", "attention_mask", "token_type_ids"])

    # finally, we can loop through and compute predictions

    data_loader: DataLoader = DataLoader(tokenized_train_set, batch_size=1)  # simple cooperation with torch DataLoader!
    n_correct: int = 0
    for model_input, item in zip(data_loader, small_train_set):
        model_output = model(**model_input)
        prediction: int = torch.argmax(logits.squeeze()).item()
        label: int = item['label']
        n_correct += prediction == label and 1 or 0

    print(f"Accuracy: {n_correct / len(small_train_set)}")



