from fastapi import FastAPI
from pydantic import BaseModel

from app.preprocess import clean_text
from app.dummy_model import DummyModel, DummyVectorizer


app = FastAPI()

model = DummyModel()
vectorizer = DummyVectorizer()

THRESHOLD = 0.5


class InputText(BaseModel):
    text: str


@app.post("/predict")
def predict_api(input: InputText):

    # clean
    cleaned_text = clean_text(input.text)

    # vectorize
    vec = vectorizer.transform([cleaned_text])

    # predict
    scores = model.predict_proba(vec)

    # extract labels
    labels = []

    for label, score in scores.items():
        if score > THRESHOLD:
            labels.append(label)

    return {
        "labels": labels,
        "scores": scores
    }