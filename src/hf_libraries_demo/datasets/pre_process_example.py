from typing import Any, Dict, Callable

# Just a place-holder type hint to label a data point in a dataset, e.g. train_dataset[0]
from datasets import DatasetDict, load_dataset

DataPoint = Dict[str, Any]


def lowercase_text(item: DataPoint) -> DataPoint:
    """
    Convert the text of the data point to lower case

    :param item: a datapoint (needs a text attribute)
    :return:
    """
    # notice! you can just operate on the attribute you care about, and not worry about deleting/preserving the other
    # attributes. This is due to the update semantics of datasets.map
    return {"text": item['text'].lower()}

    # also acceptable: update AND return, e.g:
    # item['text'] = item['text'].lower()
    # return item


def pre_process_dataset(dataset: DatasetDict) -> DatasetDict:
    """
    Our complete pre-processing pipeline, returning the result as a new DatasetDict
    
    1. lower-case all input texts and replace new-lines with spaces
    2. remove training points labelled as part of the `misc` group in the training data (e.g. for zero-shot transfer)
    
    :param dataset: un-processed dataset
    :return: processed dataset
    """

    # Calling map on a DatasetDict applies to all splits
    # datasets do not update in place, allowing you to have several versions. Means you must assign result
    dataset = dataset.map(lowercase_text)
    assert 'label' in dataset['train'][0]  # note: non-destructive operation

    # you can also just pass a lambda. We'll do this for the new-lines replacement. 'desc' is a parameter passed for
    # tqdm, which updates labels on the progress bar.
    dataset = dataset.map(
        lambda item: {'text': ' '.join(item['text'].split())},
        desc="normalizing all white space to a single space"
    )

    # next, use filter to remove un-wanted classes in the training set only. Takes a callable which returns a boolean
    # indicating whether the item should be kept
    is_not_misc: Callable[[DataPoint], bool] = lambda item: not item['label_text'].startswith("misc")
    dataset['train'] = dataset['train'].filter(is_not_misc, desc="removing misc items")
    
    return dataset


if __name__ == '__main__':
    # loading the dataset will download if not present in your cache, or simply load from there
    dataset: DatasetDict = load_dataset("SetFit/20_newsgroups")

    # see function above for details
    dataset = pre_process_dataset(dataset)

    # notice the reduced size due to filter
    print(dataset)

