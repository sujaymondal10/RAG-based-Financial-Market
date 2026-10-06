import time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import CLASSES, IMG_SIZE, MODEL_PATH
from model import ChartCNN

class ChartPredictor:
    def __init__(self, model_path=MODEL_PATH):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt=torch.load(model_path,map_location=self.device)
        self.classes=ckpt.get("classes",CLASSES)
        self.model=ChartCNN(len(self.classes)).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.transform=transforms.Compose([
            transforms.Resize((IMG_SIZE,IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])
        ])

    @torch.inference_mode()
    def predict(self,image):
        start=time.perf_counter()
        x=self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs=torch.softmax(self.model(x),dim=1)[0].cpu().numpy()
        idx=int(np.argmax(probs))
        return {
            "signal":self.classes[idx],
            "confidence":float(probs[idx]),
            "probabilities":{c:float(p) for c,p in zip(self.classes,probs)},
            "latency_seconds":time.perf_counter()-start
        }
