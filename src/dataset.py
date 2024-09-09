import os
import json
import torch
import numpy as np
from typing import List, Optional
from transformers import AutoTokenizer

from .config import Config
from .utils import span_to_label, span_list_to_dict, pack_instances, unpack_instances

MASKED_LB_ID = -100

from transformers import DataCollatorForTokenClassification


class Batch:
    """
    A batch of data instances, each is initialized with a dict with attribute names as keys
    """

    def __init__(self, **kwargs):
        self.size = 0
        super().__init__()
        self._tensor_members = dict()
        for k, v in kwargs.items():
            if k == "batch_size":
                self.size = v
            setattr(self, k, v)
            self.register_tensor_members(k, v)

    def __getitem__(self, key):
        return getattr(self, key)

    def as_dict(self):
        return {k: getattr(self, k) for k in self._tensor_members.keys()}

    def register_tensor_members(self, k, v):
        """
        Register tensor members to the batch
        """
        if isinstance(v, torch.Tensor) or callable(getattr(v, "to", None)):
            self._tensor_members[k] = v

    def to(self, device):
        """
        Move all tensor members to the target device
        """
        for k, v in self._tensor_members.items():
            setattr(self, k, v.to(device))
        return self

    def __len__(self):
        return (
            len(tuple(self._tensor_members.values())[0]) if not self.size else self.size
        )


class DataCollator(DataCollatorForTokenClassification):
    def __call__(self, instance_list: list[dict]):
        tk_ids, attn_masks, lbs = unpack_instances(
            instance_list, ["bert_tk_ids", "bert_attn_masks", "bert_lbs"]
        )

        # Update `tk_ids`, `attn_masks`, and `lbs` to match the maximum length of the batch.
        # The updated type of the three variables should be `torch.int64``.
        # Hint: some functions and variables you may want to use: `self.tokenizer.pad()`, `self.label_pad_token_id`.

        padded_inputs = self.tokenizer.pad(
            {"input_ids": tk_ids, "attention_mask": attn_masks}
        )
        tk_ids = torch.tensor(padded_inputs.input_ids, dtype=torch.int64)
        attn_masks = torch.tensor(padded_inputs.attention_mask, dtype=torch.int64)

        max_len = tk_ids.shape[1]

        # `padding_side` is right for distilbert
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            lbs = torch.stack(
                [
                    torch.cat(
                        (lb, torch.full((max_len - len(lb),), self.label_pad_token_id)),
                        dim=0,
                    )
                    for lb in lbs
                ]
            )
        else:
            lbs = torch.stack(
                [
                    torch.cat(
                        (torch.full((max_len - len(lb),), self.label_pad_token_id), lb),
                        dim=0,
                    )
                    for lb in lbs
                ]
            )

        return Batch(input_ids=tk_ids, attention_mask=attn_masks, labels=lbs)


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        logger,
        text: Optional[List[List[str]]] = None,
        lbs: Optional[List[List[str]]] = None,
    ):
        super().__init__()
        self._text = text
        self._lbs = lbs
        self._logger = logger
        self._token_ids = None
        self._attn_masks = None
        self._bert_lbs = None

        self._partition = None

        self.data_instances = None

    @property
    def n_insts(self):
        return len(self._text)

    @property
    def text(self):
        return self._text if self._text else list()

    @property
    def lbs(self):
        return self._lbs if self._lbs else list()

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        return self.data_instances[idx]

    def prepare(self, config: Config, partition: str):
        """
        Load data from disk

        Parameters
        ----------
        config: configurations
        partition: dataset partition; in [train, valid, test, inference]

        Returns
        -------
        self
        """
        assert partition in ["train", "valid", "test", "inference"], ValueError(
            f"Argument `partition` should be one of 'train', 'valid', 'test' or 'inference!"
        )
        self._partition = partition

        file_path = os.path.normpath(os.path.join(config.data_dir, f"{partition}.json"))
        self._logger.info(f"Loading data file: {file_path}")
        self._text, self._lbs = load_data_from_json(file_path)

        self._logger.info("Encoding sequences...")
        self.encode(
            config.bert_model_name_or_path,
            {lb: idx for idx, lb in enumerate(config.bio_label_types)},
        )

        self._logger.info(f"Data loaded.")

        self.data_instances = pack_instances(
            bert_tk_ids=self._token_ids,
            bert_attn_masks=self._attn_masks,
            bert_lbs=self._bert_lbs,
        )
        return self

    def encode(self, tokenizer_name: str, lb2idx: dict):
        """
        Build BERT token masks as model input

        Parameters
        ----------
        tokenizer_name: str
            the name of the assigned Huggingface tokenizer
        lb2idx: dict
            a dictionary that maps the str labels to indices

        Returns
        -------
        self
        """
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)
        tokenized_text = tokenizer(
            self._text, add_special_tokens=True, is_split_into_words=True
        )

        self._token_ids = tokenized_text.input_ids
        self._attn_masks = tokenized_text.attention_mask

        bert_lbs_list = list()

        # Update label sequence to match the BERT tokenization.
        # The labels that are not involved in the loss calculation should be masked out by `MASKED_LB_ID`.
        # Hint: labels corresponding to [CLS], [SEP], and non-first subword tokens should be masked out.
        # You should store the updated label sequence in the `bert_lbs_list` variable.

        for idx, (bert_tk_idx_list, lbs) in enumerate(zip(self._token_ids, self._lbs)):
            word_ids = tokenized_text.word_ids(idx)

            word_ids_shifted_left = np.asarray([-100] + word_ids[:-1])
            word_ids = np.asarray(word_ids)
            is_first_wordpiece = (word_ids_shifted_left != word_ids) & (
                word_ids != None
            )

            bert_lbs = torch.full((len(bert_tk_idx_list),), -100)
            bert_lbs[is_first_wordpiece] = torch.tensor([lb2idx[lb] for lb in lbs])
            bert_lbs_list.append(bert_lbs)

        for tks, lbs in zip(self._token_ids, bert_lbs_list):
            assert len(tks) == len(lbs), ValueError(
                f"Length of token ids ({len(tks)}) and labels ({len(lbs)}) mismatch!"
            )

        self._bert_lbs = bert_lbs_list

        return self


def load_data_from_json(file_dir: str):
    """
    Load data stored in the current data format.

    Parameters
    ----------
    file_dir: str
        file directory

    """
    with open(file_dir, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    tk_seqs = list()
    lbs_list = list()

    for inst in data_list:
        # get tokens
        tk_seqs.append(inst["text"])

        # get true labels
        lbs = span_to_label(span_list_to_dict(inst["label"]), inst["text"])
        lbs_list.append(lbs)

    return tk_seqs, lbs_list
