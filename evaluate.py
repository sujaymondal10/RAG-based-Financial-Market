import pandas as pd, torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from config import LABELS_PATH, MODEL_PATH, CLASSES
from dataset import ChartDataset
from model import ChartCNN

def main():
    df=pd.read_csv(LABELS_PATH)
    _,test=train_test_split(df,test_size=.2,random_state=42,stratify=df["label"])
    loader=DataLoader(ChartDataset(test),batch_size=32)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt=torch.load(MODEL_PATH,map_location=device)
    model=ChartCNN(len(CLASSES)).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    yt=[]; yp=[]
    with torch.inference_mode():
        for x,y in loader:
            yp += model(x.to(device)).argmax(1).cpu().tolist()
            yt += y.tolist()
    print("Accuracy:",accuracy_score(yt,yp))
    print(classification_report(yt,yp,target_names=CLASSES,digits=4))
    print("Confusion matrix:\n",confusion_matrix(yt,yp))

if __name__=="__main__": main()
