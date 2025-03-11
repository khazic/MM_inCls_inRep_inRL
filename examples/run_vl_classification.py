#!/usr/bin/env python
# coding=utf-8

import json

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import datasets
import evaluate
import numpy as np
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

import transformers
from joblib import Parallel, delayed
from sklearn.metrics import classification_report
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed, AutoProcessor,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.models.qwen2_vl import Qwen2VLProcessor
import sys
sys.path.append('../')
from multimodal_search.utils.util import get_first_and_second
from multimodal_search.modeling.modeling_qwen2_vl_classification import Qwen2VLForClassification
from multimodal_search.data.tool import VLClassificationDataCollatorWithPadding, mm_preprocess, PreprocessDataset
logger = logging.getLogger(__name__)

@dataclass
class DataTrainingArguments:
    task_type: Optional[str] = field(
        default='first',
        metadata={
            "help": (
                "First-level label and second-level label"
            )
        },
    )
    remove_splits: Optional[str] = field(
        default=None,
        metadata={"help": "The splits to remove from the dataset. Multiple splits should be separated by commas."},
    )
    remove_columns: Optional[str] = field(
        default=None,
        metadata={"help": "The columns to remove from the dataset. Multiple columns should be separated by commas."},
    )
    label_file: Optional[str] = field(
        default='',
        metadata={
            "help": (
                "The name of the label column in the input dataset or a CSV/JSON file. "
                'If not specified, will use the "label" column for single/multi-label classification task'
            )
        },
    )
    max_seq_length: int = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=16,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    shuffle_train_dataset: bool = field(
        default=False, metadata={"help": "Whether to shuffle the train dataset or not."}
    )
    shuffle_seed: int = field(
        default=42, metadata={"help": "Random seed that will be used to shuffle the train dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    metric_name: Optional[str] = field(default=None, metadata={"help": "The metric to use for evaluation."})
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default='../cache/',
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=True,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


def load_label_info(path: str = '../data/firstandsecond.txt', task_type: str = 'first') -> Dict[str, Any]:
    """Load and process label information from file.
    
    Args:
        path: Path to label file, supports .txt and .json formats
        task_type: Type of labels to load ('first' or 'second' level)
        
    Returns:
        Dict containing:
            - label_type: Type of labels loaded
            - num_labels: Number of unique labels
            - label2id: Mapping from label names to ids
            - id2label: Mapping from ids to label names
            - label_id2name: Mapping from label ids to display names
def load_label_info(path='../data/firstandsecond.txt', task_type='first'):
    """
    Load label information
    :param path:
    :param task_type:
    :return:
    """
    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)
        
        # Assuming the format of labels.json is as you provided in the example
        if isinstance(label_data, list) and 'first' in label_data[0] and 'second' in label_data[0]:
            categories = label_data[0]['second']
            label2id = {str(i): i for i in range(len(categories))}
            id2label = {v: k for k, v in label2id.items()}
            num_labels = len(label2id)
            
            label_info = {
                'label_type': task_type,
                'num_labels': num_labels,
                'label2id': label2id,
                'id2label': id2label,
                'label_id2name': {str(i): categories[i] for i in range(len(categories))},
                'label_name2id': {categories[i]: str(i) for i in range(len(categories))}
            }
            return label_info
    
    first_id2name, first_name2id, second_id2name, second_name2id = get_first_and_second(path)
    if task_type == 'first':
        label_id2name = first_id2name
        label_name2id = first_name2id
    else:
        label_id2name = second_id2name
        label_name2id = second_name2id
    label_id_list = list(label_id2name.keys())
    label_id_list.sort()
    label2id = {label_id_list[i]: i for i in range(len(label_id_list))}
    id2label = {v:k for k,v in label2id.items()}
    logger.info(label2id)
    num_labels = len(label2id)
    label_info = {
        'label_type':task_type,
        'num_labels': num_labels,
        'label2id': label2id,
        'id2label': id2label,
        'label_id2name': label_id2name,
        'label_name2id': label_name2id
    }
    return label_info

def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16 or training_args.bf16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)
    extension = 'json'
    data_files = {}
    if '|' in data_args.train_file:
        data_files["train"] = data_args.train_file.split("|")
        data_files["validation"] = data_args.validation_file
    else:
        extension = data_args.train_file.split(".")[-1]
        data_files["train"] = data_args.train_file
        data_files["validation"] = data_args.validation_file

    dataset = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )
    logger.info('loading dataset')

    problem_type =  "multi_label_classification"

    label_info = load_label_info(data_args.label_file, task_type=data_args.task_type)

    num_labels = label_info.pop('num_labels')
    label2id = label_info['label2id']
    id2label = label_info['id2label']
    label_id2name = label_info['label_id2name']
    name2id = {label_id2name[k]:v for k,v in label2id.items()}
    logger.info(name2id)
    logger.info('loading label info')


    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        _attn_implementation="flash_attention_2",
        finetuning_task="text-classification",
        problem_type=problem_type,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=model_args.trust_remote_code,
    )



    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    Processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    torch_dtype = torch.bfloat16

    model = Qwen2VLForClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        torch_dtype=torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )

    total_train_sample = len(dataset['train'])
    total_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * training_args.world_size
    logger.info(f"{training_args.world_size}; Number of samples: {total_train_sample}; Total train batch size: {total_batch_size}")

    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    train_dataset = dataset["train"]
    if data_args.shuffle_train_dataset:
        logger.info("Shuffling the training dataset")
        train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)

    init_args = {'model_name_or_path': model_args.model_name_or_path}
    map_fn = mm_preprocess(**init_args)
    train_dataset = PreprocessDataset(dataset=train_dataset, process_fn=map_fn)

    eval_dataset = dataset["validation"]
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))


    with training_args.main_process_first(desc="eval dataset map pre-processing", local=False):
        eval_dataset = eval_dataset.map(map_fn, batched=True, num_proc=data_args.preprocessing_num_workers)

    data_collator_params = {
        'tokenizer': tokenizer,
        'processor': Processor,
        'max_length': max_seq_length,
        'vision_config': config.vision_config,
        'problem_type': config.problem_type,
        'label2id':label2id,
    }
    data_collator = VLClassificationDataCollatorWithPadding(**data_collator_params)
    metric = evaluate.load("./evaluation/f1.py", config_name="multilabel", cache_dir=model_args.cache_dir)

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        
        if config.problem_type == "single_label_classification":
            preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids, average="micro")
            r = classification_report(y_true=p.label_ids, y_pred=preds, output_dict=True, zero_division=1)
        else:
            theld = 0.4
            pred = np.greater(sigmoid(preds), theld).astype(np.int64)
            preds = np.array([np.where(p > 0, 1, 0) for p in preds])
            result = metric.compute(predictions=preds, references=p.label_ids, average="micro")
            labels = p.label_ids.astype(np.int64)
            r = classification_report(y_true=labels, y_pred=pred, output_dict=True, zero_division=1)
        
        r = [v for k, v in r.items() if 'weighted' in k]
        r = r[0]
        result.update(r)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()  
        Processor.save_pretrained(trainer.args.output_dir)
        tokenizer.save_pretrained(trainer.args.output_dir)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)



def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
