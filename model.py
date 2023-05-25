from dataclasses import dataclass
from transformers import PretrainedConfig, PreTrainedModel, AutoModelForSequenceClassification
from transformers.modeling_outputs import ModelOutput
from typing import Optional
import torch

class CustomedConfig(PretrainedConfig):
    def __init__(self,
                 encoder_model: str = "bert-base-uncased",
                 related_features_dim: int = 13,
                 num_labels: int = 1,
                 inner_dim: int = 1024,
                 **kwargs):
        super(CustomedConfig, self).__init__(num_labels=num_labels, **kwargs)
        self.encoder_model = encoder_model
        self.related_features_dim = related_features_dim
        self.inner_dim = inner_dim

@dataclass
class CustomedOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    encoder_logits: torch.FloatTensor = None

class CustomedModel(PreTrainedModel):
    def __init__(
        self,
        config: CustomedConfig
    ):
        super(CustomedModel, self).__init__(config)
        self.encoder = AutoModelForSequenceClassification.from_pretrained(self.config.encoder_model_path, num_labels=self.config.num_labels) 
        self.hidden_dim = self.config.num_labels + self.config.related_features_dim
        # TODO: Change fully connected layers.
        self.fc_layers = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_dim, self.config.inner_dim),
                torch.nn.LeakyReLU(inplace=True),
                torch.nn.Linear(self.config.inner_dim, self.config.num_labels),
            )
    
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        related_features=None,
    ):
        # input_ids.shape = (batch, max_length)
        # related_features.shape = (batch, related_features_dim)
        encoder_outputs = self.encoder(input_ids=input_ids, 
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    labels=labels)
        encoder_logits = encoder_outputs["logits"] # shape = (batch, 1)
        features = torch.cat((encoder_logits,related_features), 1)
        
        logits = self.fc_layers(features)
        
        return CustomedOutput(
            logits=logits,
            encoder_logits=encoder_logits
        )
        


