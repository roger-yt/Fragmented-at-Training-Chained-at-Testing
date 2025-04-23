import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from random import randint
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

import numpy as np
 
from transformers.data.data_collator import DataCollatorMixin, pad_without_fast_tokenizer_warning, _torch_collate_batch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


@dataclass
class My_collator(DataCollatorMixin):
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __post_init__(self):
        None

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        # print("examples:", examples)
        batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
            )
        # print("batch:", batch)
        # input_ids = batch["input_ids"][1]
        # mask = batch["label"][1]
        # print("input_ids:", self.tokenizer.decode(input_ids, skip_special_tokens=True))
        # masked = input_ids * mask + (1-mask)*self.tokenizer.pad_token_id
        # print("masked:", self.tokenizer.decode(masked, skip_special_tokens=True))
        # If special token mask has been preprocessed, pop it from the dict.
        mask = batch["label"]
        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        labels = labels * mask + (1-mask)*(-100)
        batch["labels"] = labels
        del batch["label"]
        return batch