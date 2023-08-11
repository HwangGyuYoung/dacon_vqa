import os
from collections import OrderedDict

import pandas as pd
import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
from timm.models import create_model
from transformers import XLMRobertaTokenizer

from beit3 import modeling_finetune
from dataset import VQADataset, base_path

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Hyperparameter Setting
CFG = {
    'MODEL_SIZE': 'large',
    'IMG_SIZE': 480
}

# Data Load
test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))
sample_submission = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))
test_img_path = 'image/test'

# Dataset & DataLoader
tokenizer = XLMRobertaTokenizer(os.path.join(base_path, 'models', 'beit3.spm'))
test_dataset = VQADataset(test_df, tokenizer, test_img_path, img_size=CFG["IMG_SIZE"], is_train=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8)

# Model Load
model_config = f'beit3_{CFG["MODEL_SIZE"]}_patch16_{CFG["IMG_SIZE"]}_vqav2'
model = create_model(
    model_config,
    pretrained=False,
    drop_path_rate=0.1,
    vocab_size=64010
)

tmp_weight = torch.load(os.path.join(base_path, 'models', '7_10_1e-05_large_model.pt'))
weight = OrderedDict()
for k, v in tmp_weight.items():
    weight[k[10:]] = v      # train.py 결과물 -> 10, distributed_train.py 결과물 -> 7

model.load_state_dict(weight)
model.eval()
model.to(device)

preds = []
with torch.no_grad():
    for data in tqdm(test_loader, total=len(test_loader)):
        images = data['image'].to(device)
        question = data['question'].to(device)
        padding_mask = data['padding_mask'].to(device)

        outputs = model(images, question, padding_mask)

        _, pred = outputs.max(-1)
        for x in pred:
            preds.append(test_dataset.label2ans[x])

sample_submission['answer'] = preds
sample_submission.to_csv('submission.csv', index=False)
