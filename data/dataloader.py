from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2TokenizerFast

import random
from typing import Tuple
import torch
import json
from tokenizer import TiktokenTokenizer
import lightning.pytorch as pl

class SFTPromptDataset(Dataset):

    def __init__(self, prompts: list[tuple], max_examples: int):
        super().__init__()    
        
        self.prompts = random.choices(prompts,k=max_examples)
        print("Length of prompt dataset: ", len(self.prompts))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx][0], self.prompts[idx][1], self.prompts[idx][2]

class SFTPromptDataModule(pl.LightningDataModule):
    def __init__(self, 
                train_json: str,
                test_json: str,
                batch_size: int,
                max_examples: int = None,
                tokenizer_name: str = "tiktoken/gpt2",
                sequence_length: int = 128,
                shuffle: bool = True,
                num_workers: int = 4):
        super().__init__()

        self.shuffle = shuffle
        self.num_workers = num_workers
        self.tokenizer_name = tokenizer_name
        self.sequence_length = sequence_length
        self.max_examples = max_examples
        self.batch_size = batch_size

        with open(train_json) as fp:
            train_dataset = json.load(fp)

        print("Processing the dataset with the tokenizer...")
        self.train_prompts = self._process_data(train_dataset)
    
    def _process_data(self, dataset: dict) -> torch.Tensor:

        if self.tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif self.tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif self.tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        prompts = []
        for chosen in dataset:
            cnt += 1
            prompt = chosen.split("Assistant:")[0] + "<|endoftext|>"
            tokens = tokenizer(prompt,
                                max_length=self.sequence_length,
                                padding="max_length",
                                truncation=True,
                                return_tensors="pt")

            prompts.append(
                [tokens['input_ids'], tokens['attention_mask'], torch.sum(tokens['attention_mask'])])
            
            #if self.max_examples and cnt >= self.max_examples:
            #    break
        
        print(f"Loaded {len(prompts)} tokens from {cnt} examples.")
        return prompts

    def setup(self, stage: str = None):
        self.train_dataset = SFTPromptDataset(self.train_prompts, self.max_examples)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                            batch_size=self.batch_size, 
                            shuffle=self.shuffle, 
                            num_workers=self.num_workers,
                            pin_memory=True
        )

class SFTDataset(Dataset):

    def __init__(self,
                 tokens: torch.tensor,
                 sequence_length: int
                 ):
        super().__init__()    
        
        self.tokens = tokens
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.tokens) - self.sequence_length - 2

    def __getitem__(self, idx):
        x = self.tokens[idx:idx + self.sequence_length]
        y = self.tokens[idx + 1:idx + self.sequence_length + 1]
        return x, y

class SFTDataModule(pl.LightningDataModule):
    def __init__(self, 
                train_json: str,
                test_json: str,
                batch_size: int,
                max_examples: int = None,
                tokenizer_name: str = "tiktoken/gpt2",
                sequence_length: int = 128,
                shuffle: bool = True,
                num_workers: int = 4):
        super().__init__()

        self.shuffle = shuffle
        self.num_workers = num_workers
        self.tokenizer_name = tokenizer_name
        self.sequence_length = sequence_length
        self.max_examples = max_examples
        self.batch_size = batch_size

        with open(train_json) as fp:
            train_dataset = json.load(fp)

        with open(test_json) as fp:
            test_dataset = json.load(fp)

        print("Processing the dataset with the tokenizer...")
        self.train_tokens = self._process_data(train_dataset)
        self.test_tokens = self._process_data(test_dataset)
    
    def _process_data(self, dataset: dict) -> torch.Tensor:

        if self.tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif self.tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif self.tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        tokens = []
        for chosen in dataset:
            cnt += 1
            response_text = chosen + "<|endoftext|>"
            response = tokenizer(response_text)

            tokens += response['input_ids']
            if self.max_examples and cnt >= self.max_examples:
                break
        
        tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"Loaded {len(tokens)} tokens from {cnt} examples.")
        return tokens

    def setup(self, stage: str = None):
        self.train_dataset = SFTDataset(
            self.train_tokens, self.sequence_length
        )

        self.test_dataset = SFTDataset(
            self.test_tokens, self.sequence_length
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                            batch_size=self.batch_size, 
                            shuffle=self.shuffle, 
                            num_workers=self.num_workers,
                            pin_memory=True
        )
        
    def val_dataloader(self):
        return DataLoader(self.test_dataset, 
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers,
                            pin_memory=True
                            )

class AnthropicHHRLHFDataset(Dataset):
    """
    https://huggingface.co/datasets/Anthropic/hh-rlhf#dataset-summary
    """

    def __init__(self,
                 pairs: list,
                 masks: list,
                 max_examples=None) -> None:
        super().__init__()
        self.pairs = pairs
        self.masks = masks
        self.max_examples = max_examples
        print("Dataset Loaded with N examples: ", len(self.pairs))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx], self.masks[idx]  # (2, T), (2, T)

class HHDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int,
                    max_examples: int = None,
                    tokenizer_name: str = "tiktoken/gpt2",
                    sequence_length: int = 128,
                    shuffle: bool = True,
                    num_workers: int = 4):
        super().__init__()

        self.shuffle = shuffle
        self.num_workers = num_workers
        self.tokenizer_name = tokenizer_name
        self.sequence_length = sequence_length
        self.max_examples = max_examples
        self.batch_size = batch_size

        print("Downloading or Loading the Anthropic Instruct Dataset")
        train_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
        test_dataset = load_dataset("Anthropic/hh-rlhf", split="test")

        print("Processing the dataset with the tokenizer...")
        self.train_pairs, self.train_masks = self._process_data(train_dataset)
        self.test_pairs, self.test_masks = self._process_data(test_dataset)
        
    def _process_data(self, dataset: dict) -> Tuple[list, list]:
        self.pairs = []
        self.masks = []
        
        if self.tokenizer_name == "huggingface/gpt2":
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        elif self.tokenizer_name == "huggingface/gpt2fast":
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        elif self.tokenizer_name == "tiktoken/gpt2":
            tokenizer = TiktokenTokenizer('gpt2')

        cnt = 0
        for data in dataset:
            positive = tokenizer(data["chosen"],
                                 max_length=self.sequence_length,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")
            positive_indices = positive["input_ids"]
            positive_mask = positive["attention_mask"]

            negative = tokenizer(data["rejected"],
                                 max_length=self.sequence_length,
                                 padding="max_length",
                                 truncation=True,
                                 return_tensors="pt")
            negative_indices = negative["input_ids"]
            negative_mask = negative["attention_mask"]

            self.pairs.append(
                torch.stack((positive_indices, negative_indices), dim=0))

            self.masks.append(
                torch.stack((positive_mask, negative_mask), dim=0))
            cnt += 1
            if self.max_examples and cnt >= self.max_examples:
                break

        return self.pairs, self.masks

    def setup(self, stage: str = None):

        self.train_dataset = AnthropicHHRLHFDataset(
            self.train_pairs, self.train_masks, self.max_examples 
        )

        self.test_dataset = AnthropicHHRLHFDataset(
            self.test_pairs, self.test_masks, self.max_examples 
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                            batch_size=self.batch_size, 
                            shuffle=self.shuffle, 
                            num_workers=self.num_workers,
                            pin_memory=True
        )
        
    def val_dataloader(self):
        return DataLoader(self.test_dataset, 
                            batch_size=self.batch_size,
                            shuffle=False,
                            num_workers=self.num_workers,
                            pin_memory=True
                            )

