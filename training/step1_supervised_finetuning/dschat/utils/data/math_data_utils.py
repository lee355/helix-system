import transformers
import copy
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence
import os
import json

import datasets
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
import pathlib
import dschat.utils.data.math_utils as utils
import random
from torch.utils.data import DataLoader
import math
from datasets import load_from_disk

IGNORE_INDEX = -100

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))

    # model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    attention_masks = [tokenized.attention_mask[0] for tokenized in tokenized_list]
    # Llama uses EOS as its padding token. Token IDs alone cannot distinguish
    # a real EOS in the sequence from padding added by the tokenizer.
    input_ids_lens = labels_lens = [mask.sum().item() for mask in attention_masks]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
        attention_masks=attention_masks,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    attention_masks = examples_tokenized["attention_masks"]
    for label, source_len, attention_mask in zip(
        labels, sources_tokenized["input_ids_lens"], attention_masks
    ):
        label[:source_len] = IGNORE_INDEX
        label.masked_fill_(attention_mask.eq(0), IGNORE_INDEX)
    return dict(input_ids=input_ids, labels=labels, attention_masks=attention_masks)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, data_split: float, tokenizer: transformers.PreTrainedTokenizer, template_variation: bool):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")

        if 'json' in data_path:
            with open(data_path) as f:
                list_data_dict = json.load(f)
            list_data_dict = [item for item in list_data_dict if "PoT" in item["source"]]
            max = []
            for i in  range(1, len(list_data_dict)):
                max.append(len(list_data_dict[i]['instruction'])+len(list_data_dict[i]['output']))
            max.sort()
            standard_inputlen = max[int(len(max))-1]
            list_data_dict = [item for item in list_data_dict if len(item['instruction'])+len(item['output'])<standard_inputlen]
            # if data_split > 0.5:
            #     list_data_dict = list_data_dict[:int(data_split*len(list_data_dict))]
            # else:
            #     list_data_dict = list_data_dict[int((1-data_split)*len(list_data_dict)):]
        else:
            list_data_dict = datasets.load_dataset(data_path)["train"]
        logging.warning("Formatting inputs...")
        if template_variation:
            PROMPT_DICT = random.choice(utils.PROMPT_TEMPLATE)
        else:
            PROMPT_DICT = utils.PROMPT_TEMPLATE_SINGLE
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

        sources = []
        for example in list_data_dict:
            if example.get("input", "") != "":
                sources.append(prompt_input.format_map(example))
            else:
                sources.append(prompt_no_input.format_map(example))

        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        self.sources = sources
        self.targets = targets

    def __len__(self):
        return len(self.sources)

    def naive__getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __getitem__(self, i):
        return dict(input_ids=self.sources[i], labels=self.targets[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, pipeline_parallelism: bool):
        self.tokenizer = tokenizer
        self.pipeline_parallelism = pipeline_parallelism

    def naive__call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        if self.pipeline_parallelism:
            return (
                    (input_ids, attention_mask),
                    labels
                )
        else:
            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        for instance in instances:
            source = instance['input_ids']
            target = instance['labels']
            sources.append(source)
            targets.append(target)

        data_dict = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data_dict['input_ids'], data_dict['labels']
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            data_dict["attention_masks"], batch_first=True, padding_value=0
        ).bool()
        if not labels[:, 1:].ne(IGNORE_INDEX).any():
            raise ValueError(
                "SFT batch contains no supervised next-token targets after truncation; "
                "increase max_seq_len or filter examples whose prompts fill the context."
            )
        if self.pipeline_parallelism:
            return (
                    (input_ids, attention_mask),
                    labels
                )
        else:
            return dict(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
            )

class DataCollatorForFinanceDataset(object):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, pipeline_parallelism: bool):
        self.tokenizer = tokenizer
        self.pipeline_parallelism = pipeline_parallelism
    def naive__call__(self, features: list) -> dict:
        # Check if pad_token_id is None, if it is then use eos_token_id as the padding value
        if self.tokenizer.pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id  # Use eos_token_id as a fill symbol
        else:
            pad_token_id = self.tokenizer.pad_token_id

        len_ids = [len(feature["input_ids"]) for feature in features]
        longest = max(len_ids)

        input_ids = []
        labels_list = []

        for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
            ids = feature["input_ids"]
            seq_len = feature["seq_len"]

            # Padding with calculated pad_token_id
            labels = (
                [pad_token_id] * (seq_len - 1) + ids[(seq_len - 1) :] + [pad_token_id] * (longest - ids_l)
            )
            ids = ids + [pad_token_id] * (longest - ids_l)

            _ids = torch.LongTensor(ids)
            labels_list.append(torch.LongTensor(labels))
            input_ids.append(_ids)

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels_list)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        if self.pipeline_parallelism:
            return (
                    (input_ids,attention_mask),
                    labels
                )
        else:
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask":attention_mask,
            }
    def __call__(self, features: list) -> dict:
        # Check if pad_token_id is None, if it is then use eos_token_id as the padding value
        if self.tokenizer.pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id  # Use eos_token_id as a fill symbol
        else:
            pad_token_id = self.tokenizer.pad_token_id

        len_ids = [len(feature["input_ids"]) for feature in features]
        longest = max(len_ids)

        input_ids = []
        labels_list = []

        for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
            ids = feature["input_ids"]
            seq_len = feature["seq_len"]

            # Padding with calculated pad_token_id
            labels = (
                [pad_token_id] * (seq_len - 1) + ids[(seq_len - 1) :] + [pad_token_id] * (longest - ids_l)
            )
            ids = ids + [pad_token_id] * (longest - ids_l)

            _ids = torch.LongTensor(ids)
            labels_list.append(torch.LongTensor(labels))
            input_ids.append(_ids)

        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels_list)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        if self.pipeline_parallelism:
            return (
                    (input_ids,attention_mask),
                    labels
                )
        else:
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask":attention_mask,
            }

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, args, data_split=0.1, pipeline_parallelism=False) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_split=data_split, data_path=args.data_path,
                                      template_variation=True)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, pipeline_parallelism=pipeline_parallelism)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    if pipeline_parallelism:
        return train_dataset, data_collator
    else:
        return train_dataset, data_collator, train_dataloader#dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def make_finance_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, args, data_split=0.1, pipeline_parallelism=False) -> Dict:
    train_dataset = load_from_disk(args.data_path)
    data_collator = DataCollatorForFinanceDataset(tokenizer=tokenizer, pipeline_parallelism=pipeline_parallelism)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    if pipeline_parallelism:
        return train_dataset, data_collator
    else:
        return train_dataset, data_collator, train_dataloader
