from ml.data import process_data
from ml.model import load_model, inference
import pandas as pd

project_path = "/home/sduer/D501/Deploying-a-Scalable-ML-Pipeline-with-FastAPI"
encoder = load_model(project_path + "/model/encoder.pkl")
model = load_model(project_path + "/model/model.pkl")

sample = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

data = {k: [v] for k, v in sample.items()}

print('Creating DataFrame with columns:', list(data.keys()))

df = pd.DataFrame.from_dict(data)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

try:
    X_proc, y, enc, lb = process_data(X=df, categorical_features=cat_features, training=False, encoder=encoder)
    print('Processed shape:', X_proc.shape)
    preds = inference(model, X_proc)
    print('Predictions:', preds)
except Exception as e:
    import traceback
    traceback.print_exc()
    print('Error:', e)
