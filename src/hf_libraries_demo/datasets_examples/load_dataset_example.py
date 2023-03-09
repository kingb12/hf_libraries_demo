from pprint import pprint

from datasets import load_dataset, DatasetDict, Dataset

if __name__ == '__main__':
    # loading the dataset will download if not present in your cache, or simply load from there
    dataset: DatasetDict = load_dataset("SetFit/20_newsgroups")

    # most datasets are loaded as DatasetDict's, where keys are splits. In this case we have 'train' and 'test'
    assert sorted(list(dataset.keys())) == ['test', 'train'], f"unexpected splits or keys! {dataset}"
    print(dataset)

    # you can load a particular split by using the split argument, getting a Dataset instead of DatasetDict
    train_dataset: Dataset = load_dataset("SetFit/20_newsgroups", split="train")

    # you can iterate over a dataset with a for loop, or index as a list. Generally, you can treat datasets as a List
    # of Dict objects which are loaded into memory only as needed, but also supporting map and filter operations
    # unique to the datasets library
    pprint(train_dataset[0])
    for item in train_dataset:
        # For this dataset, we have an input text of the newsgroup post ('text'), the label integer ('label') and
        # the label as a string ('label_text')
        assert 'text' in item and type(item['text']) == str
        assert 'label_text' in item and type(item['label_text']) == str
        assert 'label' in item and type(item['label']) == int

    # loading separately is the same as just accessing from DatasetDict:
    for item_a, item_b in zip(train_dataset, dataset['train']):
        assert item_a == item_b, "datasets aren't the same!"
