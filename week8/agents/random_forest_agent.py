# imports

import os
import re
from typing import List
from sentence_transformers import SentenceTransformer
import joblib
from agents.agent import Agent



class RandomForestAgent(Agent):
    """
    The RandomForestAgent uses a pre-trained Random Forest model to estimate the price of a product.
    It first converts the product description into a vector embedding using a SentenceTransformer model,
    and then feeds this vector into the Random Forest model to get a price prediction.
    """

    name = "Random Forest Agent"
    color = Agent.MAGENTA

    def __init__(self):
        """
        Initializes the RandomForestAgent.
        This involves loading the pre-trained Random Forest model and the SentenceTransformer model
        for vectorizing text descriptions.
        """
        self.log("Random Forest Agent is initializing")
        # Load the sentence transformer model for creating text embeddings.
        self.vectorizer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        # Load the pre-trained Random Forest model from a file.
        self.model = joblib.load('random_forest_model.pkl')
        self.log("Random Forest Agent is ready")

    def price(self, description: str) -> float:
        """
        Estimates the price of a product using the pre-trained Random Forest model.

        Args:
            description: The description of the product to be priced.

        Returns:
            The estimated price of the product as a float.
        """        
        self.log("Random Forest Agent is starting a prediction")
        # Convert the product description into a vector embedding.
        vector = self.vectorizer.encode([description])
        # Use the Random Forest model to predict the price from the vector.
        # The result is capped at a minimum of 0 to prevent negative prices.
        result = max(0, self.model.predict(vector)[0])
        self.log(f"Random Forest Agent completed - predicting ${result:.2f}")
        return result
