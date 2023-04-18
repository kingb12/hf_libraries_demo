"""
This file covers some examples with generating from T5 or other encoder-decoder models. Each example will use some
the Anthropic/hh-rlhf dataset, comparing to the chosen response. If you're unfamiliar, please note the prompts and thus
generations can be toxic, as the purpose of the dataset is to reduce toxicity in generative-LMs via RLHF.

In general, I'll try to include an example for calling with one prompt, and then batching.
"""
from pprint import pprint
from typing import List, Tuple, Any, Dict

import torch
from datasets import load_dataset, Dataset
from torch import Tensor
from transformers import T5Tokenizer, T5ForConditionalGeneration


def separate_prompt_and_response(
        conversation: str, task_prefix: str = "Continue the conversation as an Assistant:\n\n") -> Tuple[str, str]:
    """
    Anthropic/hh-rlhf includes 'chosen' and 'rejected' text attributes which are each a string concatenating a prompt
    and response pair. This method extracts the prompt and response from one of these attributes, where the prompt was
    given and the response was chosen or rejected, depending on calling context. So far, this method is not batch
    callable.

    :param conversation: conversation to split (e.g. 'chosen' or 'rejected' attribute from hh-rlhf data item)
    :param task_prefix: if given, inserts this before the conversation as a task-prefix in the returned prompt. Default
    is on with prefix "Continue the conversation as an Assistant:\n\n"

    :return: tuple with prompt, response
    """
    # separate prompt from chosen response
    turns: List[str] = [t for t in conversation.split("\n\n") if t]
    response: str = turns[-1]
    prompt: str = "\n\n".join(t for t in turns[:-1])
    prompt = task_prefix + prompt
    return prompt, response


def greedy_decode_one(prompt: str, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer) -> str:
    """
    Generates a single response with greedy decoding

    :param prompt: input prompt, given to encoder
    :param model: T5 model variant (it should have the same generation interface as T5)
    :param tokenizer: tokenizer appropriate for model
    :return: generated response string, with special tokens removed
    """
    # choose whether to run on the GPU or on CPU
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_ids: Tensor = tokenizer(prompt, return_tensors="pt")['input_ids'].to(device)
    outputs: Tensor = model.generate(
        inputs=input_ids,  # don't need to worry about attention masks, default is fully visible
    )
    # we generate a batch tensor, which will just have one element (batch_size=1)
    # skipping special tokens cleans up the EOS token (</s>) and any extra generated padding, etc.
    response: str = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def sample_from_one(prompt: str, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer,
                    num_return_sequences: int = 8) -> List[str]:
    """
    Generates a single response with greedy decoding

    :param prompt: input prompt, given to encoder
    :param model: T5 model variant (it should have the same generation interface as T5)
    :param tokenizer: tokenizer appropriate for model
    :param num_return_sequences: how many sequences to sample (length of returned list)
    :return: generated response strings in a list, with special tokens removed
    """
    # choose whether to run on the GPU or on CPU
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_ids: Tensor = tokenizer(prompt, return_tensors="pt")['input_ids'].to(device)
    outputs: Tensor = model.generate(
        inputs=input_ids,  # don't need to worry about attention masks, default is fully visible
        do_sample=True,
        top_k=0,  # might be default, but manually prevents top-k from being used
        temperature=0.7,
        num_return_sequences=num_return_sequences,
    )
    # we generate a batch tensor, which will have num_return_sequences elements (the number of sequences we sampled)
    # skipping special tokens cleans up the EOS token (</s>) and any extra generated padding, etc.
    responses: List[str] = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return responses


def sample_from_one_with_prefix(prompt: str, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer,
                                decoder_prefix: str, num_return_sequences: int = 8) -> List[str]:
    """
    Generates a single response with sampling and a decoder prefix

    :param prompt: input prompt, given to encoder
    :param model: T5 model variant (it should have the same generation interface as T5)
    :param tokenizer: tokenizer appropriate for model
    :param decoder_prefix: string prefix to start each decoding with. Can significantly control generation
    :param num_return_sequences: how many sequences to sample (length of returned list)
    :return: generated response strings in a list, with special tokens removed
    """
    # choose whether to run on the GPU or on CPU
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prepare decoder prefix as a tensor
    # need to skip special tokens or it will add </s>, messing up insertion as decoder_input_ids
    decoder_inputs = tokenizer(decoder_prefix, return_tensors="pt", add_special_tokens=False)

    # insert the decoder start token id as first element b/c apparently this argument removes it
    decoder_input_ids: Tensor = torch.cat([
        torch.tensor([[model.config.decoder_start_token_id]]),  # note batch of 1
        decoder_inputs.input_ids
    ], dim=1).to(model.device)

    # prepare inputs as normal
    input_ids: Tensor = tokenizer(prompt, return_tensors="pt")['input_ids'].to(device)

    # sample with additional argument
    outputs: Tensor = model.generate(
        inputs=input_ids,  # don't need to worry about attention masks, default is fully visible
        do_sample=True,
        top_k=0,  # might be default, but manually prevents top-k from being used
        temperature=0.7,
        decoder_input_ids=decoder_input_ids,
        num_return_sequences=num_return_sequences,
    )

    # we generate a batch tensor, which will have num_return_sequences elements (the number of sequences we sampled)
    # skipping special tokens cleans up the EOS token (</s>) and any extra generated padding, etc.
    responses: List[str] = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return responses


def greedy_decode_batch(dataset: Dataset, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer) -> Dataset:
    """
    Generates a 't5_greedy_decode_response' attribute in the dataset based on T5 response

    :param dataset: containing 'prompt' attribute
    :param model: T5 model variant (it should have the same generation interface as T5)
    :param tokenizer: tokenizer appropriate for model
    :return: dataset with t5_greedy_decode_response included
    """
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def handle_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        input_ids: Tensor = tokenizer(
            batch['prompt'],
            return_tensors="pt",
            padding=True,
            max_length=model.config.max_length,
            truncation=True
        )['input_ids'].to(device)
        outputs: Tensor = model.generate(
            inputs=input_ids,  # don't need to worry about attention masks, default is fully visible
        )
        responses: List[str] = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return {"t5_greedy_decode_response": responses}  # gets merged into dictionary via map()

    # call handle batch via map
    dataset = dataset.map(handle_batch, batched=True, batch_size=64, desc="greedy decoding via batches")
    return dataset


def sample_from_batch(dataset: Dataset, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer,
                      num_return_sequences_per_item: int = 8) -> Dataset:
    """
    Generates a 't5_sampled_responses' attribute in the dataset based on T5 response from sampling, which is a list of
    num_return_sequences_per_item sampled responses

    :param dataset: containing 'prompt' attribute
    :param model: T5 model variant (it should have the same generation interface as T5)
    :param tokenizer: tokenizer appropriate for model
    :return: dataset with t5_sampled_responses included
    """
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def handle_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        input_ids: Tensor = tokenizer(
            batch['prompt'],
            return_tensors="pt",
            padding=True,
            max_length=model.config.max_length,
            truncation=True
        )['input_ids'].to(device)
        outputs: Tensor = model.generate(
            inputs=input_ids,  # don't need to worry about attention masks, default is fully visible,
            do_sample=True,
            top_k=0,  # might be default, but manually prevents top-k from being used
            temperature=0.7,
            num_return_sequences=num_return_sequences_per_item,
        )

        responses: List[str] = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # we received (batch_size * num_return_sequences_per_item) responses in a single list, need to
        # transform to a list of lists where there are batch_size list's of num_return_sequences_per_item items
        # (e.g. 512 -> 64 x 8)
        responses: List[List[str]] = [responses[i:i + num_return_sequences_per_item]
                                      # iterate in chunks of size num_return_sequences_per_item, no remainders
                                      for i in range(0, len(responses), num_return_sequences_per_item)]
        return {"t5_sampled_response": responses}  # gets merged into dictionary via map()

    # call handle batch via map
    dataset = dataset.map(handle_batch, batched=True, batch_size=64, desc="sampling via batches")
    return dataset


def sample_from_batch_with_prefix(dataset: Dataset, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer,
                                  decoder_prefix: str, num_return_sequences_per_item: int = 8) -> Dataset:
    """
    Generates a 't5_sampled_responses' attribute in the dataset based on T5 response from sampling, which is a list of
    num_return_sequences_per_item sampled responses

    :param dataset: containing 'prompt' attribute
    :param model: T5 model variant (it should have the same generation interface as T5)
    :param tokenizer: tokenizer appropriate for model
    :param decoder_prefix: string to start each decoding with
    :return: dataset with t5_sampled_responses included
    """
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def handle_batch(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        # prepare decoder prefix as a tensor
        # need to skip special tokens or it will add </s>, messing up insertion as decoder_input_ids
        decoder_inputs = tokenizer(decoder_prefix, return_tensors="pt", add_special_tokens=False)

        # insert the decoder start token id as first element b/c apparently this argument removes it
        decoder_input_ids: Tensor = torch.cat([
            torch.tensor([[model.config.decoder_start_token_id]]),  # note batch of 1
            decoder_inputs.input_ids
        ], dim=1).to(model.device)

        # expand it to batch size to repeat for each element in batch!
        batch_size: int = len(batch['prompt'])
        decoder_input_ids = decoder_input_ids.expand((batch_size, -1))

        # prepare inputs as normal
        input_ids: Tensor = tokenizer(
            batch['prompt'],
            return_tensors="pt",
            padding=True,
            max_length=model.config.max_length,
            truncation=True
        )['input_ids'].to(device)

        # call with additional argument
        outputs: Tensor = model.generate(
            inputs=input_ids,  # don't need to worry about attention masks, default is fully visible,
            do_sample=True,
            top_k=0,  # might be default, but manually prevents top-k from being used
            temperature=0.7,
            num_return_sequences=num_return_sequences_per_item,
            decoder_input_ids=decoder_input_ids
        )

        responses: List[str] = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # we received (batch_size * num_return_sequences_per_item) responses in a single list, need to
        # transform to a list of lists where there are batch_size list's of num_return_sequences_per_item items
        # (e.g. 512 -> 64 x 8)
        responses: List[List[str]] = [responses[i:i + num_return_sequences_per_item]
                                      # iterate in chunks of size num_return_sequences_per_item, no remainders
                                      for i in range(0, len(responses), num_return_sequences_per_item)]
        return {"t5_sampled_response": responses}  # gets merged into dictionary via map()

    # call handle batch via map
    dataset = dataset.map(handle_batch, batched=True, batch_size=64, desc="sampling via batches")
    return dataset

if __name__ == '__main__':
    # We'll work with just 256 examples to speed up performance, batching demonstrations, etc.
    dataset = load_dataset('Anthropic/hh-rlhf')
    dataset: Dataset = dataset['train'].select(range(4222, 4222 + 256))
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)

    # Example 1: just generating a single response, from greedy text
    prompt, gold_response = separate_prompt_and_response(dataset[0]['chosen'])
    t5_response = greedy_decode_one(prompt, model=model, tokenizer=tokenizer)
    print(t5_response)  # Note: starts with Human even though it's Assistant's turn


    # Before batching experiments, add prompt and gold_response attributes
    def prompt_response_dict(item: Dict[str, Any]) -> Dict[str, Any]:
        # needed to return a dict in map()
        prompt, response = separate_prompt_and_response(item['chosen'])
        return {"prompt": prompt, "response": response}


    dataset = dataset.map(prompt_response_dict, desc="adding task prefixes and extracting prompts")

    # Example 2: Greedy decoding as a batch
    dataset = greedy_decode_batch(dataset=dataset, model=model, tokenizer=tokenizer)

    # Example 3: multinomial sampling with temperature = 0.7
    prompt, gold_response = separate_prompt_and_response(dataset[0]['chosen'])
    t5_responses: List[str] = sample_from_one(prompt, model=model, tokenizer=tokenizer)
    pprint(t5_responses)  # still all starting with Human

    # Example 4: multinomial sampling w/ t=0.7, batched
    dataset = sample_from_batch(dataset=dataset, model=model, tokenizer=tokenizer)

    # Problem: so far, we can't seem to get the model to turn-take effectively, generating Human as a prefix too often
    # and repeating the turn. Further, single turns are not obvious. To improve things, we'll include an "Assistant: "
    # prefix at generation time, but in the decoder. Decoder prefixes have been shown to be more effective than further
    # encoder-side inputs in controlling generation (see PET-Gen paper: https://arxiv.org/pdf/2012.11926.pdf)

    # Example 5: multinomial sampling with temperature = 0.7 with a decoder prefix
    prompt, gold_response = separate_prompt_and_response(dataset[0]['chosen'])
    t5_responses: List[str] = sample_from_one_with_prefix(prompt, model=model, tokenizer=tokenizer,
                                                          decoder_prefix="Assistant:")
    pprint(t5_responses)  # Guaranteed to start with "Assistant:"

    # Example 6: multinomial sampling with temperature = 0.7 and a decoder prefix, batched
    dataset = sample_from_batch_with_prefix(dataset=dataset, model=model, tokenizer=tokenizer,
                                            decoder_prefix="Assistant:")  # Guaranteed to start with "Assistant:"

