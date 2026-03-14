import pytest
# TODO: add necessary import
import os
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

def test_process_data_returns_preprocessing_objects_on_training():
    """
    Returns the LabelBinarizer and OneHotEncoder objects from ml.data.process_data() when training = True
    """
    
    data = {"a": 1, "b": 2, "c": 3}
    df = pd.DataFrame(data, index=[0])
    c_features = ["b", "c"]

    _, _, encoder, lb = process_data(
        X=df,
        categorical_features=c_features,
        label="a",
        training=True
    )

    assert isinstance(encoder, OneHotEncoder) and isinstance(lb, LabelBinarizer)

def test_can_save_model():
    """
    Checks that we can save a model to the filesystem
    """
    
    model = RandomForestClassifier()
    model_path = os.path.join(os.getcwd(), "tes_model.pkl")
    save_model(model, model_path)

    assert os.path.exists(model_path)

    # cleanup
    if os.path.exists(model_path):
        os.remove(model_path)

def test_can_load_model():
    """
    Checks that we can load a model from the file system.

    NB: dependent on test_can_save_model()
    """
    model = RandomForestClassifier()
    model_path = os.path.join(os.getcwd(), "tes_model.pkl")
    save_model(model, model_path)

    model_loaded = load_model(model_path)
    assert isinstance(model_loaded, RandomForestClassifier)

    # cleanup
    if os.path.exists(model_path):
        os.remove(model_path)
