# 03_SourceCode/credit_utils.py (bản đầy đủ)
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_COLS = ["LOAN", "MORTDUE", "VALUE", "YOJ", "DEROG", "DELINQ", "CLAGE", "NINQ", "CLNO", "DEBTINC"]
CAT_COLS = ["REASON", "JOB"]
ENG_COLS = ["LTV_EXISTING", "LTV_COMBINED", "EQUITY", "DEROG_DELINQ", "INQ_PER_LINE"]
SEED = 42


def load_and_split(path="../data/hmeq.csv", seed=SEED):
    """Chia train/test đúng một nơi duy nhất để mọi notebook dùng chung."""
    df = pd.read_csv(path)
    X, y = df.drop(columns="BAD"), df["BAD"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)

class CreditFeatures(BaseEstimator, TransformerMixin):
    """Tạo 5 đặc trưng nghiệp vụ tín dụng. Không học gì từ dữ liệu nên không gây rò rỉ."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()                                   # không sửa đè lên dữ liệu gốc
        X["LTV_EXISTING"] = X["MORTDUE"] / X["VALUE"]
        X["LTV_COMBINED"] = (X["MORTDUE"] + X["LOAN"]) / X["VALUE"]
        X["EQUITY"] = X["VALUE"] - X["MORTDUE"]
        X["DEROG_DELINQ"] = X[["DEROG", "DELINQ"]].sum(axis=1, min_count=1)
        X["INQ_PER_LINE"] = X["NINQ"] / (X["CLNO"] + 1)
        return X.replace([np.inf, -np.inf], np.nan)    # chia cho 0 thì coi như thiếu

class Winsorizer(BaseEstimator, TransformerMixin):
    """Cắt ngoại lai theo percentile học từ tập train (không xoá dòng nào)."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lo_ = X.quantile(self.lower)      # mốc sàn, học từ dữ liệu fit
        self.hi_ = X.quantile(self.upper)      # mốc trần
        return self

    def transform(self, X):
        return pd.DataFrame(X).clip(self.lo_, self.hi_, axis=1).to_numpy()

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)    # để sau này lấy được tên cột

def make_preprocessor(scale, use_fe=True, add_indicator=True):
    num_cols = NUM_COLS + (ENG_COLS if use_fe else [])
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=add_indicator))]
    if scale:
        num_steps = [("winsor", Winsorizer())] + num_steps + [("scale", StandardScaler())]
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", Pipeline(num_steps), num_cols), ("cat", cat_pipe, CAT_COLS)])


def build_pipeline(model, scale, use_fe=True, add_indicator=True):
    steps = [("fe", CreditFeatures())] if use_fe else []
    steps += [("prep", make_preprocessor(scale, use_fe, add_indicator)), ("clf", model)]
    return Pipeline(steps)