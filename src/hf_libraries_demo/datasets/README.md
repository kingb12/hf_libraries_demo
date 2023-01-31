# Huggingface Datasets Examples

I recommend reading through the [quickstart](https://huggingface.co/docs/datasets/quickstart) and docs at Huggingface, 
many of these examples come directly from this source.

In this directory are examples of basic dataset operations, each grouped into a file or directory as appropriate.

## Loading a Dataset

For these examples, we'll use the classification dataset 
[20-newsgroups](https://huggingface.co/datasets/SetFit/20_newsgroups)

> The 20 newsgroups dataset comprises around 18000 newsgroups posts on 20 topics split in two subsets: one for training (or development) and the other one for testing (or for performance evaluation). The split between the train and test set is based upon a messages posted before and after a specific date.

See the [Quickstart](https://huggingface.co/docs/datasets/quickstart) for more examples, as well as audio and vision datasets

Example in [`load_dataset_example.py`](./load_dataset_example.py)

### Useful Environment Variables for `nlp-gpu-01`

See 
Both `transformers` and `datasets` cache downloaded data and models, and can quickly overrun your home directory in 
`/soe`. Add these environment variables to your run-time (customize to your user + location preference):

```bash
HF_HOME=/data/users/<username>/.cache/huggingface
TRANSFORMERS_CACHE=/data/users/<username>/.cache/huggingface
```