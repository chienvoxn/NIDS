import torch
from torch import nn

class FeatureTokenizer(nn.Module):
 def __init__(self,n_features,d_model):
  super().__init__()
  self.weight=nn.Parameter(torch.empty(n_features,d_model))
  self.bias=nn.Parameter(torch.empty(n_features,d_model))
  nn.init.normal_(self.weight,mean=0.,std=.02)
  nn.init.normal_(self.bias,mean=0.,std=.02)
 def forward(self,x):
  return x.unsqueeze(-1)*self.weight.unsqueeze(0)+self.bias.unsqueeze(0)

class SEChannels(nn.Module):
 def __init__(self,d_model,reduction=8):
  super().__init__()
  hidden=max(8,d_model//reduction)
  self.net=nn.Sequential(nn.Linear(d_model,hidden),nn.GELU(),nn.Linear(hidden,d_model),nn.Sigmoid())
 def forward(self,z):
  # z: B,F,D ; squeeze over feature-token positions
  return z * self.net(z.mean(dim=1)).unsqueeze(1)

class TokenResBlock(nn.Module):
 def __init__(self,d_model,dropout):
  super().__init__()
  self.norm=nn.LayerNorm(d_model)
  self.net=nn.Sequential(nn.Linear(d_model,2*d_model),nn.GELU(),nn.Dropout(dropout),nn.Linear(2*d_model,d_model))
  self.dropout=nn.Dropout(dropout)
 def forward(self,x):return x+self.dropout(self.net(self.norm(x)))

class TokenizedCNNSEAttention(nn.Module):
 def __init__(self,n_features,n_classes,d_model=64,heads=4,dropout=.15,
              use_cnn=True,use_se=True,use_attn=True,use_raw=False):
  super().__init__()
  assert d_model%heads==0
  self.flags={'use_cnn':use_cnn,'use_se':use_se,'use_attn':use_attn,'use_raw':use_raw}
  self.tokenizer=FeatureTokenizer(n_features,d_model)
  self.norm0=nn.Identity()  # preserve scalar feature magnitude before feature mixing
  if use_cnn:
   self.convs=nn.ModuleList([nn.Conv1d(d_model,d_model,k,padding=k//2) for k in (1,3,5)])
   self.conv_fuse=nn.Sequential(nn.Conv1d(3*d_model,d_model,1),nn.GELU())
   self.norm_cnn=nn.LayerNorm(d_model)
  self.se=SEChannels(d_model) if use_se else nn.Identity()
  if use_attn:
   self.attn_norm=nn.LayerNorm(d_model)
   self.mha=nn.MultiheadAttention(d_model,heads,dropout=dropout,batch_first=True)
   self.attn_dropout=nn.Dropout(dropout)
  self.token_mlp=TokenResBlock(d_model,dropout)
  self.pool_gate=nn.Linear(d_model,1)
  rep_dim=3*d_model
  if use_raw:
   self.raw_mlp=nn.Sequential(nn.Linear(n_features,2*d_model),nn.LayerNorm(2*d_model),nn.GELU(),
                              nn.Dropout(dropout),nn.Linear(2*d_model,d_model),nn.GELU())
   rep_dim+=d_model
  self.classifier=nn.Sequential(nn.LayerNorm(rep_dim),nn.Linear(rep_dim,2*d_model),nn.GELU(),
                                nn.Dropout(dropout),nn.Linear(2*d_model,n_classes))
 def forward(self,x):
  z=self.norm0(self.tokenizer(x))
  if self.flags['use_cnn']:
   q=z.transpose(1,2)
   mixed=self.conv_fuse(torch.cat([conv(q) for conv in self.convs],dim=1)).transpose(1,2)
   z=self.norm_cnn(z+mixed)
  z=self.se(z)
  if self.flags['use_attn']:
   q=self.attn_norm(z)
   z=z+self.attn_dropout(self.mha(q,q,q,need_weights=False)[0])
  z=self.token_mlp(z)
  weights=torch.softmax(self.pool_gate(z).squeeze(-1),dim=1)
  attentive=(z*weights.unsqueeze(-1)).sum(dim=1)
  reps=[attentive,z.mean(dim=1),z.amax(dim=1)]
  if self.flags['use_raw']:reps.append(self.raw_mlp(x))
  return self.classifier(torch.cat(reps,dim=1))
