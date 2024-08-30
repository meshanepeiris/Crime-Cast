import torch
import pandas as pd

#load pre-trained bert model
model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')

#load the csv to fine tune
# pd.read('../csv/news')