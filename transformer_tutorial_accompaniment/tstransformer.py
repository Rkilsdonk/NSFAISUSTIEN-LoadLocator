from .mha import MultiHeadAttention
from .utils import gen_batch, jagged_to_padded, benchmark
from .te_layer import TransformerEncoderLayer
from .td_layer import TransformerDecoderLayer
from .transformer import Transformer, TransformerDecoder, TransformerEncoder
import torch
import torch.nn as nn
import math
from typing import Optional
import copy as copy
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)
logging.basicConfig(
     filename="tstransformer.log",
     encoding="utf-8",
     filemode="a",
     format="{asctime} - {levelname} - {message}",
     style="{",
     datefmt="%Y-%m-%d %H:%M"
 )

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:,  0::2] = torch.sin(position * div_term)
        pe[:,  1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
class TsTransformer(nn.Module):
    def __init__(self, d_input: int,d_output:int,d_latent:int,numblocks:int,nheads:int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.project=nn.Linear(in_features=d_input,out_features=d_latent)
        self.encode=PositionalEncoding(d_latent)
        self.mainblocks=nn.ModuleList()
        for i in range(numblocks):
            self.mainblocks.append(nn.TransformerEncoderLayer(d_model=d_latent,nhead=nheads,dim_feedforward=d_latent,dropout=dropout))
        self.decode=nn.LazyLinear(d_output)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output=self.project(x) #Mulitply x by key 1 to get it to size, should have shape [T*embedding]
        output=self.encode(output) #Add dimensions
        mask=nn.Transformer.generate_square_subsequent_mask(output.size(0))
        for block in self.mainblocks:
            output=block(output,src_mask=mask,is_causal=True)
        output.squeeze(1)
        
        #block ordering is normalize, attention, normalize, feedforwards
        output=self.decode(x)#return to shape,
        return output
class LSTMGS(nn.Module):
    def __init__(self, d_input: int,d_output:int,d_latent:int,numblocks:int):
        super().__init__()
        self.LSTM=nn.LSTM(input_size=d_input,hidden_size=d_latent, num_layers=numblocks)
        self.h_0=nn.Parameter(torch.randn([numblocks,d_latent])/(numblocks*d_latent))
        self.c_0=nn.Parameter(torch.randn([numblocks,d_latent])/(numblocks*d_latent))
        self.downproject=nn.Linear(in_features=d_latent,out_features=d_output)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output=self.LSTM(x,(self.h_0,self.c_0))
        output=self.downproject(output[0])
        return output
class ActiveTransformer(nn.Module):
    def __init__(self, d_input: int,d_output:int,d_latent:int,numblocks:int,nheads:int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.project=nn.Linear(in_features=d_input,out_features=d_latent)
        self.encode=PositionalEncoding(d_latent)
        self.mainblocks=nn.ModuleList()
        for i in range(numblocks):
            self.mainblocks.append(nn.TransformerEncoderLayer(d_model=d_latent,nhead=nheads,dim_feedforward=d_latent,dropout=dropout))
        self.decode=nn.LazyLinear(d_output)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output=self.project(x) #Mulitply x by key 1 to get it to size, should have shape [T*embedding]
        output=self.encode(output) #Add dimensions
        mask=nn.Transformer.generate_square_subsequent_mask(output.size(0))
        for block in self.mainblocks:
            output=block(output,src_mask=mask,is_causal=True)
        output.squeeze(1)
        
        #block ordering is normalize, attention, normalize, feedforwards
        output=self.decode(x)#return to shape,
        return output