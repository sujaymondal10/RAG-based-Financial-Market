import argparse, time
import numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from config import LABELS_PATH, ROOT
from predict import ChartPredictor

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=100)
    n=ap.parse_args().n
    df=pd.read_csv(LABELS_PATH).head(n)
    p=ChartPredictor(); times=[]
    for rel in df["path"]:
        img=Image.open(ROOT/rel)
        times.append(p.predict(img)["latency_seconds"])
    a=np.array(times)
    print("Samples:",len(a))
    print("Mean latency:",a.mean(),"sec")
    print("P95 latency:",np.percentile(a,95),"sec")
    print("Estimated throughput:",3600/a.mean(),"charts/hour")

if __name__=="__main__": main()
