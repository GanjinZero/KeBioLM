import csv
import glob
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import tqdm

from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputExample:
    example_id: str
    sent0: str
    sent1: str
    label: object

@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    example_id: str
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    first_entity_position: List[int]
    second_entity_position: List[int]
    label: object

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


if is_torch_available():
    import torch
    from torch.utils.data.dataset import Dataset

    class RelationExtractionDataset(Dataset):
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
        ):
            task = task.lower()
            if task in ["chemprot", "ddi"]:
                ignore_first = False
            if task in ["gad"]:
                ignore_first = True

            self.processor = REProcessor(ignore_first)
            processor = self.processor

            cached_features_file = os.path.join(
                data_dir,
                "cached_{}_{}_{}_{}_{}".format(
                    mode.value,
                    tokenizer.__class__.__name__,
                    str(max_seq_length),
                    task,
                    str(len(tokenizer))
                ),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")
                    if mode == Split.dev:
                        examples = processor.get_dev_examples(data_dir)
                    elif mode == Split.test:
                        examples = processor.get_test_examples(data_dir)
                    else:
                        examples = processor.get_train_examples(data_dir)
                    logger.info("Training examples: %s", len(examples))
                    label_list = processor.get_labels()
                    marker_list = processor.marker_list
                    self.features = convert_examples_to_features(
                        examples,
                        label_list,
                        max_seq_length,
                        tokenizer,
                        marker_list
                    )
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


if is_tf_available():
    import tensorflow as tf

    class TFRelationExtractionDataset:
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            max_seq_length: Optional[int] = 128,
            overwrite_cache=False,
            mode: Split = Split.train,
        ):
            processor = processors[task]()

            logger.info(f"Creating features from dataset file at {data_dir}")
            label_list = processor.get_labels()
            if mode == Split.dev:
                examples = processor.get_dev_examples(data_dir)
            elif mode == Split.test:
                examples = processor.get_test_examples(data_dir)
            else:
                examples = processor.get_train_examples(data_dir)
            logger.info("Training examples: %s", len(examples))

            self.features = convert_examples_to_features(
                examples,
                label_list,
                max_seq_length,
                tokenizer,
            )

            def gen():
                for (ex_index, ex) in tqdm.tqdm(enumerate(self.features), desc="convert examples to features"):
                    if ex_index % 10000 == 0:
                        logger.info("Writing example %d of %d" % (ex_index, len(examples)))

                    yield (
                        {
                            "example_id": 0,
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "token_type_ids": ex.token_type_ids,
                        },
                        ex.label,
                    )

            self.dataset = tf.data.Dataset.from_generator(
                gen,
                (
                    {
                        "example_id": tf.int32,
                        "input_ids": tf.int32,
                        "attention_mask": tf.int32,
                        "token_type_ids": tf.int32,
                    },
                    tf.int64,
                ),
                (
                    {
                        "example_id": tf.TensorShape([]),
                        "input_ids": tf.TensorShape([None, None]),
                        "attention_mask": tf.TensorShape([None, None]),
                        "token_type_ids": tf.TensorShape([None, None]),
                    },
                    tf.TensorShape([]),
                ),
            )

        def get_dataset(self):
            self.dataset = self.dataset.apply(tf.data.experimental.assert_cardinality(len(self.features)))

            return self.dataset

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def _read_tsv(self, path, ignore_first=True):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if ignore_first:
            lines = lines[1:]
        return [line.strip().split("\t") for line in lines]

    def _find_marker(self, lines):
        marker_set = set()
        for line in lines:
            tmp = ""
            for ch in " ".join(line):
                if ch == "@":
                    tmp = "@"
                elif ch == "$":
                    tmp = tmp + "$"
                    marker_set.update([tmp])
                    tmp = ""
                else:
                    tmp = tmp + ch

        marker_list = list(marker_set)
        marker_list.sort()
        return marker_list

    def _find_labels(self, lines):
        labels = list(set([l[-1].strip() for l in lines]))
        if 'false' in labels:
            labels.remove('false')
            labels.sort()
            labels = ['false'] + labels
        elif 'DDI-false' in labels:
            labels.remove('DDI-false')
            labels.sort()
            labels = ['DDI-false'] + labels
        else:
            labels.sort()
        return labels


class REProcessor(DataProcessor):
    def __init__(self, ignore_first):
        self.marker_list = None
        self.labels = None
        self.ignore_first = ignore_first

    def get_train_examples(self, data_dir):
        logger.info("LOOKING AT {} train".format(data_dir))
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"), ignore_first=self.ignore_first)
        return self._create_examples(lines, "train")

    def get_dev_examples(self, data_dir):
        logger.info("LOOKING AT {} dev".format(data_dir))
        lines = self._read_tsv(os.path.join(data_dir, "dev.tsv"), ignore_first=self.ignore_first)
        return self._create_examples(lines, "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        lines = self._read_tsv(os.path.join(data_dir, "test.tsv"), ignore_first=self.ignore_first)
        return self._create_examples(lines, "test")

    def get_labels(self):
        return self.labels

    def _create_examples(self, lines, set_type):
        if self.marker_list is None:
            self.marker_list = self._find_marker(lines)
        if self.labels is None:
            self.labels = self._find_labels(lines)

        examples = []
        for idx, line in enumerate(lines):
            race_id = line[0].strip() + "-" + set_type
            text0 = line[1]
            label = line[-1].strip()
            examples.append(InputExample(example_id=race_id, sent0=text0, sent1=None, label=label))
        return examples

def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    marker_list: List[str],
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """
    label_map = {label: i for i, label in enumerate(label_list)}

    tokenizer.add_tokens(marker_list)
    marker_id_list = []
    for marker in marker_list:
        marker_id = tokenizer.encode(marker, add_special_tokens=False)
        assert len(marker_id) == 1
        marker_id_list.append(marker_id[0])

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        text_a = example.sent0
        text_b = example.sent1

        inputs = tokenizer(
            text_a,
            text_b,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_overflowing_tokens=True,
        )

        first_entity_position = None
        second_entity_position = None
        for idx, input_id in enumerate(inputs["input_ids"]):
            if input_id in marker_id_list:
                if first_entity_position is None:
                    first_entity_position = idx
                else:
                    second_entity_position = idx
                    break

        if first_entity_position is None:
            first_entity_position = 0

        if second_entity_position is None:
            second_entity_position = first_entity_position

        features.append(
            InputFeatures(
                example_id=example.example_id,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"],
                first_entity_position=first_entity_position,
                second_entity_position=second_entity_position,
                label=label_map[example.label],
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features

