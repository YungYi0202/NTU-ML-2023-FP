from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
import os

@dataclass
class CustomedOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    encoder_logits: torch.FloatTensor = None

class CustomedModel(nn.Module):
    def __init__(
        self,
        encoder_name_or_path=None,
        num_classes=1,
        related_features_dim=14,
        fc_layers=[1024]
    ):
        super(CustomedModel, self).__init__()
        self.num_classes = num_classes
        self.related_features_dim = related_features_dim
        self.fc_layers_size = fc_layers
        if encoder_name_or_path is not None:
            self._init(encoder_name_or_path)
        
    def _init(self, encoder_name_or_path):
        self.encoder = AutoModelForSequenceClassification.from_pretrained(encoder_name_or_path, num_labels=self.num_classes)
        self.hidden_size = len(self.encoder.config.id2label) + self.related_features_dim
        
        self.fc_layers = nn.ModuleList()
        prev_size = self.hidden_size
        
        # Add fully connected layers
        for fc_size in self.fc_layers_size:
            self.fc_layers.append(nn.Linear(prev_size, fc_size))
            self.fc_layers.append(nn.ReLU())
            prev_size = fc_size
        
        self.fc_layers.append(nn.Linear(prev_size, self.num_classes))

    def forward(
        self,
        input_ids,
        related_features,
        token_type_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        # input_ids.shape = (batch, max_length)
        # related_features.shape = (batch, related_features_dim)
        encoder_outputs = self.encoder(input_ids=input_ids, 
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    labels=labels)
        encoder_logits = encoder_outputs["logits"] # shape = (batch, 1)
        logits = torch.cat((encoder_logits,related_features), 1)
        
        for layer in self.fc_layers:
            logits = layer(logits)
        
        return CustomedOutput(
            logits=logits,
            encoder_logits=encoder_logits
        )

    def from_pretrained(self, 
        checkpoint, 
        num_classes=1,
        related_features_dim=14,
        fc_layers=[1024]):
        self.num_classes = num_classes
        self.related_features_dim = related_features_dim
        self.fc_layers_size = fc_layers
        self._init(os.path.join(checkpoint, "encoder"))
        self.fc_layers.load_state_dict(torch.load(os.path.join(checkpoint, "fc_layers.bin")))

        return self
