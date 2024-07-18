import numpy as np
import pandas as pd

# Sample data for product reviews
reviews = [
    "This product is amazing! I love it.",
    "Decent quality, but a bit overpriced.",
    "Terrible experience. Would not recommend.",
    "Great value for money. Very satisfied.",
    "The product arrived damaged. Disappointed.",
    "Excellent customer service and fast shipping.",
    "Average product, nothing special.",
    "Exceeded my expectations. Will buy again!",
    "Poor quality control. Returned immediately.",
    "Good product overall, but could be improved."
]

ratings = np.array([5, 3, 1, 5, 2, 5, 3, 5, 1, 4], dtype=np.float32)

# Create a pandas DataFrame
df = pd.DataFrame({
    'review_text': reviews,
    'rating': ratings
})

SURREAL_DEAL_REVIEWS = {
    "reviews": df['review_text'].tolist(),
    "ratings": df['rating'].tolist(),
    "dataframe": df
}