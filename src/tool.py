import json
import logging
from dataclasses import dataclass
from typing import Union, Optional, List, Any, Dict

from torch.utils.data import Dataset
from torch import Tensor
from transformers import PreTrainedTokenizerBase, BatchFeature, Qwen2VLProcessor
from transformers.utils import PaddingStrategy
from transformers import AutoTokenizer, AutoConfig, AutoProcessor
from typing import (
    List,
    Optional,
    Union,
)
import torch
from vision_process import process_vision_info
logger = logging.getLogger(__name__)

@dataclass
class VLClassificationDataCollatorWithPadding:
    """
    """
    vision_config: dict
    tokenizer: PreTrainedTokenizerBase
    processor: Qwen2VLProcessor
    padding: Union[bool, str, PaddingStrategy] = "max_length"
    max_length: Optional[int] = 1024
    pad_to_multiple_of: Optional[int] = 8
    return_tensors: str = "pt"
    label2id: Optional[dict[str, int]] = None
    problem_type:str = "multi_label_classification" # "regression" "multi_label_classification" "single_label_classification"
    def __call__(self, features: List[Dict[str, Any]]) -> BatchFeature:
        batch = {
                'messages':[],
                'text':[],
                'label':[]
        }
        for feature_item in features:
            message = json.loads(feature_item['message']) if type(feature_item['message']) == str else feature_item['message']
            batch['messages'].append(message)
            batch['text'].append(feature_item['text'])
            batch['label'].append(feature_item['label'])
        label = self.process_labels(batch['label'])
        max_length = self.max_length
        texts = batch['text']
        messages = batch['messages']
        image_inputs, video_inputs = process_vision_info(messages)
        batch_items = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            max_length=max_length,
            verbose=False,
        )
        batch_items['labels'] = label

        return batch_items


    def process_labels(self, labels):
        if self.problem_type == "regression":
            return torch.tensor(labels, dtype=torch.float)
        elif self.problem_type == "single_label_classification":
            return torch.tensor(labels, dtype=torch.long)
        else:
            return self.labels_to_ids(labels)


    def labels_to_ids(self, labels_list: List[dict]) -> Tensor:
        batch_labels = []
        for labels in labels_list:
            ids = [0.0] * len(self.label2id)  # BCELoss requires float as target type
            for label in labels:
                if label in self.label2id:
                    ids[self.label2id[label]] = 1.0
            batch_labels.append(ids)
        batch_labels = torch.tensor(batch_labels, dtype=torch.float)
        return batch_labels


class PreprocessDataset(Dataset):
    def __init__(self, dataset, process_fn=None, **kwargs):
        self.process_fn = process_fn
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item_id):
        return self.process_fn(self.dataset[item_id])

class mm_preprocess:
    def __init__(self, model_name_or_path,
                 max_length=1024,):
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.vision_config = self.config.vision_config
        self.max_length = max_length
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

    def __call__(self, item):
        message = json.loads(item['message']) if type(item['message']) == str else item['message']
        input_text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=False
        )
        result = {
            'message':item['message'],
            'text':input_text,
        }
        if 'label' in item:
            result['label'] = item['label']
        return result

    def map(self, examples):
        input_item = [json.loads(message) if type(message) == str else message for message in examples['message'] ]
        input_text = self.processor.apply_chat_template(
            input_item, tokenize=False, add_generation_prompt=False
        )
        result = {
            'message':examples['message'],
            'text':input_text,
        }
        if 'label' in examples:
            result['label'] = examples['label']
        return result
