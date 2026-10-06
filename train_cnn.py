import argparse, random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from config import LABELS_PATH, MODEL_PATH, CLASSES
from dataset import ChartDataset
from model import ChartCNN

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def epoch(model, loader, criterion, optimizer, device, training):
    model.train(training)
    losses=[]; yt=[]; yp=[]
    for x,y in loader:
        x,y=x.to(device),y.to(device)
        with torch.set_grad_enabled(training):
            logits=model(x); loss=criterion(logits,y)
            if training:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(loss.item())
        yt += y.detach().cpu().tolist()
        yp += logits.argmax(1).detach().cpu().tolist()
    return np.mean(losses), accuracy_score(yt,yp), f1_score(yt,yp,average="macro")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=15)
    ap.add_argument("--batch-size",type=int,default=32)
    ap.add_argument("--lr",type=float,default=1e-3)
    args=ap.parse_args()
    seed_everything()

    df=pd.read_csv(LABELS_PATH)
    if len(df)<30: raise RuntimeError("Too few samples; run make_dataset.py first.")
    tr,va=train_test_split(df,test_size=.2,random_state=42,stratify=df["label"])
    train=DataLoader(ChartDataset(tr,True),batch_size=args.batch_size,shuffle=True)
    valid=DataLoader(ChartDataset(va),batch_size=args.batch_size)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=ChartCNN(len(CLASSES)).to(device)
    counts=tr["label"].value_counts()
    weights=torch.tensor([len(tr)/(len(CLASSES)*counts.get(c,1))
                          for c in CLASSES],dtype=torch.float32,device=device)
    criterion=nn.CrossEntropyLoss(weight=weights)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)

    best=-1
    for e in range(1,args.epochs+1):
        a=epoch(model,train,criterion,opt,device,True)
        b=epoch(model,valid,criterion,opt,device,False)
        print(f"Epoch {e:02d} | train loss/acc/f1={a} | val loss/acc/f1={b}")
        if b[2]>best:
            best=b[2]
            torch.save({"model_state":model.state_dict(),
                        "classes":CLASSES,"img_size":224},MODEL_PATH)
    print("Best validation macro-F1:",best)
    print("Saved:",MODEL_PATH)

if __name__=="__main__": main()
