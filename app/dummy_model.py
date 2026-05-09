class DummyVectorizer:
    def transform(self, texts):
        return texts


class DummyModel:

    LABELS = ["toxic", "insult", "threat", "obscene"]

    def predict_proba(self, X):

        text = X[0]

        scores = {
            "toxic": 0.1,
            "insult": 0.1,
            "threat": 0.1,
            "obscene": 0.1
        }

        if "hate" in text:
            scores["toxic"] = 0.91

        if "stupid" in text:
            scores["insult"] = 0.88

        if "kill" in text:
            scores["threat"] = 0.95

        if "fuck" in text:
            scores["obscene"] = 0.93

        return scores