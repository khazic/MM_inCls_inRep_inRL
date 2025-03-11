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
from multimodal_search.utils.vision_process import process_vision_info
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
            batch['messages'].append(feature_item['messages'])
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
            # 对于CIFAR10，标签可能是字符串形式的数字，需要转换为整数
            return torch.tensor([int(label) for label in labels], dtype=torch.long)
        else:
            return self.labels_to_ids(labels)


    def labels_to_ids(self, labels_list: List[dict]) -> Tensor:
        batch_labels = []
        for label in labels_list:
            # 处理可能的字符串标签
            if isinstance(label, str):
                if label in self.label2id:
                    ids = [0.0] * len(self.label2id)
                    ids[self.label2id[label]] = 1.0
                    batch_labels.append(ids)
            else:
                ids = [0.0] * len(self.label2id)  # BCELoss requires float as target type
                for l in label:
                    if l in self.label2id:
                        ids[self.label2id[l]] = 1.0
                batch_labels.append(ids)
        batch_labels = torch.tensor(batch_labels, dtype=torch.float)
        return batch_labels


class PreprocessDataset(Dataset):
    def __init__(self, dataset, process_fn=None, **kwargs):
        self.dataset = dataset
        self.process_fn = process_fn
        if self.process_fn is not None:
            self.processed_data = [self.process_fn(self.dataset[i]) for i in range(len(self.dataset))]
        else:
            self.processed_data = self.dataset

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, item_id):
        return self.processed_data[item_id]

class mm_preprocess:
    def __init__(self, model_name_or_path,
                 max_length=1024,):
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.vision_config = self.config.vision_config
        self.max_length = max_length
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)

    def __call__(self, examples):
        messages = examples['message']
        input_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        result = {
            'messages': messages,
            'text': input_text,
        }
        if 'label' in examples:
            result['label'] = examples['label']
        return result
