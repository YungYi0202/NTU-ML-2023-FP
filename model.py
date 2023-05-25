from dataclasses import dataclass
from transformers import PretrainedConfig, PreTrainedModel, AutoModelForSequenceClassification
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput
from typing import List
import torch

class CustomedConfig(PretrainedConfig):
    def __init__(self,
                 encoder_model: str = "bert-base-uncased",
                 num_labels: int = 1,
                 inner_dim: int = 1024,
                 **kwargs):
        super(CustomedConfig, self).__init__(num_labels=num_labels, **kwargs)
        self.encoder_model = encoder_model
        self.inner_dim = inner_dim

@dataclass
class CustomedOutput(Seq2SeqSequenceClassifierOutput):
    # loss: Optional[torch.FloatTensor] = None
    # logits: torch.FloatTensor = None
    # past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    # encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    # encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_logits: torch.FloatTensor = None

class CustomedModel(PreTrainedModel):
    def __init__(
        self,
        config: CustomedConfig
    ):
        super(CustomedModel, self).__init__(config)
        self.feature_names = ['Energy','Key','Loudness','Speechiness',
            'Acousticness','Instrumentalness','Liveness','Valence','Tempo',
            'Duration_ms','Views','Likes','Stream']
    
        self.encoder = AutoModelForSequenceClassification.from_pretrained(self.config.encoder_model_path, num_labels=self.config.num_labels) 
        self.hidden_dim = self.config.num_labels + len(self.feature_names)
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
        Energy=None,
        Key=None,
    ):
        # data.shape = (batch, max_length)
        return self.encoder(input_ids=input_ids, 
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    labels=labels)
        


