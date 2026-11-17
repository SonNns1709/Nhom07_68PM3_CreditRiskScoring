# 03_SourceCode/credit_utils.py (bản tối thiểu)
import pandas as pd
from sklearn.model_selection import train_test_split

NUM_COLS = ["LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"]
CAT_COLS = ["REASON", "JOB"]
SEED = 42


def load_and_split(path="../data/hmeq.csv", seed=SEED):
    """Chia train/test đúng một nơi duy nhất để mọi notebook dùng chung."""
    df = pd.read_csv(path)
    X, y = df.drop(columns="BAD"), df["BAD"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)