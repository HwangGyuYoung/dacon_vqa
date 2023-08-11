from dataclasses import dataclass
import datetime
import os
import random
from tqdm.auto import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers import XLMRobertaTokenizer, get_cosine_schedule_with_warmup
from timm.models import create_model

from beit3 import utils, modeling_finetune
from dataset import VQADataset, base_path

import warnings
warnings.filterwarnings(action='ignore')

# Hyperparameter Setting
CFG = {
    'MODEL_SIZE': 'large',
    'IMG_SIZE': 480,
    'EPOCHS': 10,
    'LEARNING_RATE': 1e-5,
    'BATCH_SIZE': 8,
    'SEED': 41
}


@dataclass
class DistributedArgs:
    world_size: int = 4
    gpu: tuple = (0, 1, 2, 3)
    dist_url: str = 'tcp://0.0.0.0:37860'
    dist_backend: str = 'nccl'


# Fixed RandomSeed
random.seed(CFG['SEED'])
os.environ['PYTHONHASHSEED'] = str(CFG['SEED'])
np.random.seed(CFG['SEED'])
torch.manual_seed(CFG['SEED'])
torch.cuda.manual_seed(CFG['SEED'])
torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = True
torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    args = DistributedArgs()
    mp.spawn(main_worker, args=(args,), nprocs=args.world_size, join=True)


def main_worker(rank, args):
    torch.cuda.set_device(args.gpu[rank])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=rank,
        timeout=datetime.timedelta(0, 7200)
    )
    torch.distributed.barrier()

    # Data Load
    train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
    train_img_path = 'image/train'

    # dataset & dataloader
    tokenizer = XLMRobertaTokenizer(os.path.join(base_path, 'models', 'beit3.spm'))
    train_dataset = VQADataset(train_df, tokenizer, train_img_path, img_size=CFG['IMG_SIZE'], is_train=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], num_workers=4, sampler=train_sampler, pin_memory=True)

    # Model Load
    model_config = f'beit3_{CFG["MODEL_SIZE"]}_patch16_{CFG["IMG_SIZE"]}_vqav2'
    model = create_model(
        model_config,
        pretrained=False,
        drop_path_rate=0.4,
        vocab_size=64010
    )

    utils.load_model_and_may_interpolate(
        ckpt_path=os.path.join(base_path, 'models', f'beit3_{CFG["MODEL_SIZE"]}_indomain_patch16_224.zip'),
        model=model,
        model_key='model|module',
        model_prefix=''
    )
    model.to(device)
    model = DDP(model, device_ids=[args.gpu[rank]], find_unused_parameters=True)

    # Train
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=CFG["LEARNING_RATE"], betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=len(train_loader) * int(CFG["EPOCHS"] * 0.1),
        num_training_steps=len(train_loader) * CFG["EPOCHS"]
    )
    scaler = torch.cuda.amp.GradScaler()

    model.train()
    for epoch in range(1, CFG['EPOCHS']+1):
        total_loss = 0

        for data in tqdm(train_loader, total=len(train_loader)):
            images = data['image'].to(device)
            question = data['question'].to(device)
            padding_mask = data['padding_mask'].to(device)
            answer = data['answer'].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images, question, padding_mask)
                loss = criterion(input=outputs.float(), target=answer.float())
            total_loss += loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

        if rank == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch}/{CFG["EPOCHS"]}], Train Loss: [{avg_loss:.5f}]')

            torch.save(
                model.state_dict(),
                os.path.join(
                    base_path, 'models',
                    f'{epoch}_{CFG["EPOCHS"]}_{"{:.0e}".format(CFG["LEARNING_RATE"])}_large_model.pt'
                )
            )


if __name__ == '__main__':
    main()
