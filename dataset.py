import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import CLASSES, IMG_SIZE, ROOT

class ChartDataset(Dataset):
    def __init__(self, frame, augment=False):
        self.frame = frame.reset_index(drop=True)
        ops = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
        if augment:
            ops += [transforms.RandomHorizontalFlip(p=0.2),
                    transforms.RandomRotation(3)]
        ops += [
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ]
        self.transform = transforms.Compose(ops)
        self.label_to_id = {c:i for i,c in enumerate(CLASSES)}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        image = Image.open(ROOT / row["path"]).convert("RGB")
        return self.transform(image), self.label_to_id[row["label"]]
