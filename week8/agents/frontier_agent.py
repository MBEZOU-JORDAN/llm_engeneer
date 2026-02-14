# imports

import os
import re
import math
import json
from typing import List, Dict
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import chromadb
from items import Item
from testing import Tester
from agents.agent import Agent


class FrontierAgent(Agent):
    """
    The FrontierAgent estimates the price of a product by leveraging a large language model (LLM)
    like GPT-4o-mini or DeepSeek. It enhances the LLM's performance by using a Retrieval-Augmented
    Generation (RAG) approach. This involves finding similar products in a Chroma vector datastore
    and providing them as context in the prompt sent to the LLM.
    """

    name = "Frontier Agent"
    color = Agent.BLUE

    MODEL = "gpt-4o-mini"
    
    def __init__(self, collection):
        """
        Initializes the FrontierAgent.
        This sets up the connection to the LLM (either OpenAI or DeepSeek, depending on available API keys),
        connects to the Chroma datastore, and initializes the sentence transformer model for vector encoding.

        Args:
            collection: The ChromaDB collection to be used for RAG.
        """
        self.log("Initializing Frontier Agent")
        # Check for DeepSeek API key and configure the client accordingly.
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        if deepseek_api_key:
            self.client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
            self.MODEL = "deepseek-chat"
            self.log("Frontier Agent is set up with DeepSeek")
        else:
            # Default to OpenAI if DeepSeek key is not found.
            self.client = OpenAI()
            self.MODEL = "gpt-4o-mini"
            self.log("Frontier Agent is setting up with OpenAI")
        self.collection = collection
        # Load the sentence transformer model for creating text embeddings.
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.log("Frontier Agent is ready")

    def make_context(self, similars: List[str], prices: List[float]) -> str:
        """
        Creates a context string to be inserted into the LLM prompt.
        This string contains descriptions and prices of similar products.

        Args:
            similars: A list of descriptions of similar products.
            prices: A list of prices for the similar products.

        Returns:
            A formatted string containing the context of similar products.
        """
        message = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
        for similar, price in zip(similars, prices):
            message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
        return message

    def messages_for(self, description: str, similars: List[str], prices: List[float]) -> List[Dict[str, str]]:
        """
        Creates the list of messages to be sent to the LLM API.
        This includes the system message, user prompt with context, and an assistant prefix.

        Args:
            description: The description of the product to be priced.
            similars: A list of descriptions of similar products.
            prices: A list of prices for the similar products.

        Returns:
            A list of messages in the format expected by the OpenAI API.
        """
        system_message = "You estimate prices of items. Reply only with the price, no explanation"
        user_prompt = self.make_context(similars, prices)
        user_prompt += "And now the question for you:\n\n"
        user_prompt += "How much does this cost?\n\n" + description
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "Price is $"}
        ]

    def find_similars(self, description: str):
        """
        Finds and returns items similar to the given description by querying the Chroma datastore.

        Args:
            description: The description of the product.

        Returns:
            A tuple containing two lists: the descriptions of similar documents and their prices.
        """
        self.log("Frontier Agent is performing a RAG search of the Chroma datastore to find 5 similar products")
        # Encode the description into a vector.
        vector = self.model.encode([description])
        # Query the Chroma collection for the 5 most similar items.
        results = self.collection.query(query_embeddings=vector.astype(float).tolist(), n_results=5)
        documents = results['documents'][0][:]
        prices = [m['price'] for m in results['metadatas'][0][:]]
        self.log("Frontier Agent has found similar products")
        return documents, prices

    def get_price(self, s: str) -> float:
        """
        A utility function to extract a floating-point number from a string.

        Args:
            s: The string to extract the price from.

        Returns:
            The extracted price as a float, or 0.0 if no price is found.
        """
        s = s.replace('
        , '').replace(',', '')
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0.0

    def price(self, description: str) -> float:
        """
        Estimates the price of a product by calling the configured LLM.
        It first finds similar products to provide context and then makes the API call.

        Args:
            description: The description of the product to be priced.

        Returns:
            The estimated price of the product.
        """
        # Find similar products for RAG context.
        documents, prices = self.find_similars(description)
        self.log(f"Frontier Agent is about to call {self.MODEL} with context including 5 similar products")
        # Call the LLM with the generated prompt.
        response = self.client.chat.completions.create(
            model=self.MODEL, 
            messages=self.messages_for(description, documents, prices),
            seed=42,
            max_tokens=5
        )
        reply = response.choices[0].message.content
        # Extract the price from the LLM's response.
        result = self.get_price(reply)
        self.log(f"Frontier Agent completed - predicting ${result:.2f}")
        return result
        