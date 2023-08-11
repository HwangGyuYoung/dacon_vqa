import os
import random
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformers import XLMRobertaTokenizer, get_cosine_schedule_with_warmup
from timm.models import create_model

from beit3 import utils, modeling_finetune
from dataset import VQADataset, base_path

import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

# Hyperparameter Setting
CFG = {
    'IMG_SIZE': 480,
    'EPOCHS': 10,
    'LEARNING_RATE': 1e-5,
    'BATCH_SIZE': 16,
    'SEED': 41
}

# Fixed RandomSeed
random.seed(CFG['SEED'])
os.environ['PYTHONHASHSEED'] = str(CFG['SEED'])
np.random.seed(CFG['SEED'])
torch.manual_seed(CFG['SEED'])
torch.cuda.manual_seed(CFG['SEED'])
torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = True

# Data Load
train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
train_img_path = 'image/train'

# dataset & dataloader
tokenizer = XLMRobertaTokenizer(os.path.join(base_path, 'models', 'beit3.spm'))
train_dataset = VQADataset(train_df, tokenizer, train_img_path, img_size=CFG['IMG_SIZE'], is_train=True)
train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=8)

# Model Load
model_config = 'beit3_large_patch16_480_vqav2'
model = create_model(
    model_config,
    pretrained=False,
    drop_path_rate=0.4,
    vocab_size=64010
)

utils.load_model_and_may_interpolate(
    ckpt_path=os.path.join(base_path, 'models', 'beit3_large_indomain_patch16_224.zip'),
    model=model,
    model_key='model|module',
    model_prefix=''
)
model = torch.compile(model)

# Train
criterion = nn.BCEWithLogitsLoss(reduction='mean')
optimizer = torch.optim.AdamW(params=model.parameters(), lr=CFG["LEARNING_RATE"], betas=(0.9, 0.999), weight_decay=0.01)
scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=len(train_loader) * int(CFG["EPOCHS"] * 0.1),
    num_training_steps=len(train_loader) * CFG["EPOCHS"]
)

model.train()
model.to(device)
for epoch in range(1, CFG['EPOCHS']+1):
    total_loss = 0

    for data in tqdm(train_loader, total=len(train_loader)):
        images = data['image'].to(device)
        question = data['question'].to(device)
        padding_mask = data['padding_mask'].to(device)
        answer = data['answer'].to(device)

        optimizer.zero_grad()

        outputs = model(images, question, padding_mask)

        loss = criterion(input=outputs.float(), target=answer.float())
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch}/{CFG["EPOCHS"]}], Train Loss: [{avg_loss:.5f}]')

    torch.save(
        model.state_dict(),
        os.path.join(base_path, 'models', f'{epoch}_{CFG["EPOCHS"]}_{"{:.0e}".format(CFG["LEARNING_RATE"])}_large_model.pt')
    )
