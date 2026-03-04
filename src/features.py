import torch
from transformers import BertModel, BertTokenizer

from transformers import T5EncoderModel, T5Config, T5PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from torch import nn, Tensor
from typing import List, Optional, Tuple, Union

from transformers import T5EncoderModel, T5Tokenizer

import torch
import datasets
from tqdm import tqdm
import os
import json

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextFeatureExtractor():
    def __init__(self, save_path: str | None, logger=None):
        if save_path is None:
            curr_path = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(curr_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                save_path = config["feature_cache_path"]
                save_path = os.path.join(save_path, self.name)
        else:
            save_path = os.path.join(save_path, self.name)

        self.save_path = save_path
        self.logger = logger
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

        # TO BE IMPLEMENTED BY EACH MODULE
        self.features_size = None
    
    def get_feature_batch(self, img_batch: torch.Tensor):
        """TO BE IMPLEMENTED BY EACH MODULE"""
        pass
    
    def get_features_and_idxes(self, texts: datasets.Dataset, name=None, recompute=False, num_samples=5000, batchsize=128):
        """
        Gets the features from imgs (a Dataset).
        - name: Unique name of set of images for caching purposes
        - recompute: Whether to recompute cached features
        - num_samples: number of samples
        - batchsize: batch size in computing features
        """
        if self.save_path and name:
            file_path = os.path.join(self.save_path, f"{name}.pt")

            if not recompute:
                if os.path.exists(file_path):
                    load_file = torch.load(file_path)
                    if self.logger is not None:
                        self.logger.info("Found saved features and idxes: {}".format(file_path))
                    return load_file['features'], load_file['idxes']

        if isinstance(texts, datasets.Dataset):
            features, idxes = self.get_dataset_features_and_idxes(texts, num_samples, batchsize)
        else:
            raise NotImplementedError(
                f"Cannot get features from '{type(texts)}'. Expected datasets.Dataset"
            )

        if self.save_path and name:
            if self.logger is not None:
                self.logger.info("Saving features and idxes to {}".format(file_path))
            torch.save({"features": features, "idxes": idxes}, file_path)

        return features, idxes
    
    def chunk_examples(self, examples, batchsize = 64):
     chunks = []
     for paragraph in examples['text']:
         chunks += [paragraph[i:i + batchsize] for i in range(0, len(paragraph), batchsize)]
     return {'chunks': chunks}

    def get_dataset_features_and_idxes(self, dataset: datasets.Dataset, num_samples=5000, batchsize=128):
        size = min(num_samples, len(dataset))
        features = torch.zeros(size, self.features_size)
        texts = []
        
        dataset = dataset.shuffle(seed=42).select(range(size))
        
        start_idx = 0
        for i in tqdm(range(0, len(dataset), batchsize)):
            batch = dataset[i:i+batchsize]['text']
            feature = self.get_feature_batch(batch)
            
            features[start_idx : start_idx + feature.shape[0]] = feature
            texts.extend(batch)

            start_idx = start_idx + feature.shape[0]
        return features, texts

class BERTFeatureExtractor(TextFeatureExtractor):
    def __init__(self, save_path=None, logger=None, API_KEY = 'your_api_key'):
        self.name = "bert"

        super().__init__(save_path, logger)

        self.features_size = 768
        
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        self.model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def preprocess(self, text, bert_model, tokenizer):
        
        tokens = tokenizer.tokenize(text)
        
        if len(tokens) > 254:
            tokens = tokens[:254]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        T=256
        padded_tokens=tokens +['[PAD]' for _ in range(T-len(tokens))]
        attn_mask=[ 1 if token != '[PAD]' else 0 for token in padded_tokens  ]
        
        seg_ids=[0 for _ in range(len(padded_tokens))]
        
        sent_ids=tokenizer.convert_tokens_to_ids(padded_tokens)
        
        token_ids = torch.tensor(sent_ids).unsqueeze(0)
        attn_mask = torch.tensor(attn_mask).unsqueeze(0)
        seg_ids   = torch.tensor(seg_ids).unsqueeze(0)
        
        return token_ids, attn_mask, seg_ids
    
    def get_feature_batch(self, text_batch):
        token_ids_batch = None
        attn_mask_batch = None
        seg_ids_batch = None
        for text in text_batch:
            token_ids, attn_mask, seg_ids = self.preprocess(text, self.model, self.tokenizer)
            # print(token_ids.shape, attn_mask.shape, seg_ids.shape)
            if token_ids_batch is None:
                token_ids_batch = token_ids
                attn_mask_batch = attn_mask
                seg_ids_batch = seg_ids
            else:
                token_ids_batch = torch.cat((token_ids_batch, token_ids), axis=0)
                attn_mask_batch = torch.cat((attn_mask_batch, attn_mask), axis=0)
                seg_ids_batch = torch.cat((seg_ids_batch, seg_ids), axis=0)
        
        token_ids_batch = token_ids_batch.to(self.device)
        attn_mask_batch = attn_mask_batch.to(self.device)
        seg_ids_batch = seg_ids_batch.to(self.device)
        
        with torch.no_grad():
            output = self.model(token_ids_batch, attention_mask=attn_mask_batch,token_type_ids=seg_ids_batch)
            last_hidden_state, pooler_output = output[0], output[1]

        return pooler_output

class T5ProjectionConfig(T5Config):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.project_in_dim = kwargs.get("project_in_dim", 768)
        self.project_out_dim = kwargs.get("out_dim", 4096)

class T5EncoderWithProjection(T5PreTrainedModel):
    config_class = T5ProjectionConfig

    def __init__(self, config):
        super().__init__(config)
        # self.encoder = encoder
        self.encoder = T5EncoderModel(config)
        
        self.final_projection = nn.Sequential(
            nn.Linear(config.project_in_dim, config.project_out_dim, bias=False), 
            nn.ReLU(), 
            nn.Dropout(0.1), 
            nn.Linear(config.project_out_dim, config.project_out_dim, bias=False) 
        )
        
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, T5EncoderModel

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5EncoderModel.from_pretrained("google-t5/t5-small")
        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model(input_ids=input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        return_dict = return_dict if return_dict is not None else False

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = self.final_projection(encoder_outputs[0])
        # last_hidden_state = self.final_block(last_hidden_state)[0]

        if not return_dict:
            return tuple(
                v for v in [last_hidden_state] if v is not None
            )
        
        return BaseModelOutput(
            last_hidden_state=last_hidden_state
        )


class T5FeatureExtractor(TextFeatureExtractor):
    def __init__(self, save_path=None, logger=None):
        self.name = "t5"
        super().__init__(save_path, logger)

        self.features_size = 768  # T5-base output size
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = T5EncoderModel.from_pretrained('t5-base').to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')

    def preprocess(self, text):
        # T5 tokenization
        return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    def get_feature_batch(self, text_batch):
        # Tokenize batch
        inputs = self.tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            
            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            
        return mean_pooled

class RoBERTaFeatureExtractor(TextFeatureExtractor):
    def __init__(self, save_path=None, logger=None):
        self.name = "roberta"
        super().__init__(save_path, logger)

        self.features_size = 768  # roberta-base output size
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        from transformers import RobertaModel, RobertaTokenizer

        self.model = RobertaModel.from_pretrained('roberta-base').to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def preprocess(self, text):
        return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    def get_feature_batch(self, text_batch):
        # Tokenize batch
        inputs = self.tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state
            
            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            
        return mean_pooled