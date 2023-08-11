import json
import os

import torch
from torchvision.transforms import InterpolationMode
from torchvision import transforms
from torch.utils.data import Dataset

from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from PIL import Image

from beit3.randaug import RandomAugment

base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)))


def build_transform(is_train, img_size):
    if is_train:
        t = [
            RandomResizedCropAndInterpolation(img_size, scale=(0.5, 1.0), interpolation='bicubic'),
            transforms.RandomHorizontalFlip(),
            RandomAugment(
                2, 7, isPIL=True,
                augs=[
                    'Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                    'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate'
                ]
            )
        ]

    else:
        t = [
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC)
        ]

    t += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    ]
    t = transforms.Compose(t)

    return t


class VQADataset(Dataset):
    def __init__(self, df, tokenizer, img_path, *, img_size=480, is_train=True):
        self.df = df
        self.tokenizer = tokenizer
        self.transform = build_transform(is_train, img_size)
        self.img_path = img_path
        self.is_train = is_train

        ans2label_file = os.path.join(base_path, "answer2label.txt")
        ans2label = {}
        label2ans = []
        with open(ans2label_file, mode="r", encoding="utf-8") as reader:
            for i, line in enumerate(reader):
                data = json.loads(line)
                ans = data["answer"]
                label = data["label"]
                label = int(label)
                assert label == i
                ans2label[ans] = torch.tensor(i)
                label2ans.append(ans)

        self.ans2label = ans2label
        self.label2ans = label2ans

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_name = os.path.join(base_path, self.img_path, row['image_id'] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)

        question = row['question']
        question = self.tokenizer.encode_plus(
            question,
            truncation=True,
            add_special_tokens=True,
            max_length=32,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        if self.is_train:
            answer = row['answer']
            try:
                label = self.ans2label[answer]
                one_hots = torch.nn.functional.one_hot(label, num_classes=3129)
            except KeyError:    # 3129개 이외의 클래스에 해당하는 답변 예외 처리
                one_hots = torch.tensor([0]*3129)

            return {
                'image': image.squeeze(),
                'question': question['input_ids'].squeeze(),
                'padding_mask': question['attention_mask'].squeeze().logical_not().to(int),
                'answer': one_hots.squeeze()
            }

        else:
            return {
                'image': image,
                'question': question['input_ids'].squeeze(),
                'padding_mask': question['attention_mask'].squeeze().logical_not().to(int)
            }
