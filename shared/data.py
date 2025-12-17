from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["sepal_length","sepal_width","petal_length","petal_width"]
TARGET_COLUMN = "target"

def load_iris_dataframe() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    return df.rename(columns={
        "sepal length (cm)":"sepal_length",
        "sepal width (cm)":"sepal_width",
        "petal length (cm)":"petal_length",
        "petal width (cm)":"petal_width",
    })

@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

def make_split(test_size: float=0.2, random_state: int=42) -> DatasetSplit:
    df = load_iris_dataframe()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train,X_test,y_train,y_test = train_test_split(
        X,y,test_size=test_size,random_state=random_state,stratify=y
    )
    return DatasetSplit(X_train,X_test,y_train,y_test)

def example_payload() -> Dict[str,Any]:
    return {"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}
