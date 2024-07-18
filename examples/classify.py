# # Install required dependencies
# pip install "git+https://github.com/surrealdb/surrealml#egg=surrealml[sklearn]"
# pip install requests pandas numpy
from transformers import pipeline
from surrealml import SurMlFile, Engine
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from surreal_deal_reviews import SURREAL_DEAL_REVIEWS

class HFSentimentModel(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.classifier = pipeline("sentiment-analysis")

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        results = self.classifier(list(X))
        return np.array([1 if r['label'] == 'POSITIVE' else 0 for r in results])

# Create and save the model
model = HFSentimentModel()
model.fit(SURREAL_DEAL_REVIEWS["reviews"], SURREAL_DEAL_REVIEWS["ratings"])

file = SurMlFile(
    model=model,
    name="review-sentiment-analysis",
    inputs=SURREAL_DEAL_REVIEWS["reviews"],
    engine=Engine.SKLEARN
)

file.add_version(version="0.0.1")
file.add_column("review_text")
file.add_output("sentiment", "identity")
file.save(path="./sentiment_model.surml")