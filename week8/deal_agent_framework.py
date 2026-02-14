import os
import sys
import logging
import json
from typing import List, Optional
from twilio.rest import Client
from dotenv import load_dotenv
import chromadb
from agents.planning_agent import PlanningAgent
from agents.deals import Opportunity
from sklearn.manifold import TSNE
import numpy as np


# ANSI escape codes for logging colors
BG_BLUE = '\033[44m'
WHITE = '\033[37m'
RESET = '\033[0m'

# Categories and corresponding colors for plotting
CATEGORIES = ['Appliances', 'Automotive', 'Cell_Phones_and_Accessories', 'Electronics','Musical_Instruments', 'Office_Products', 'Tools_and_Home_Improvement', 'Toys_and_Games']
COLORS = ['red', 'blue', 'brown', 'orange', 'yellow', 'green' , 'purple', 'cyan']

def init_logging():
    """
    Initializes the logging configuration for the application.
    It sets up a logger that writes to the standard output with a specific format.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [Agents] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %z",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)

class DealAgentFramework:
    """
    The main framework for the deal-finding agent system.
    It initializes and coordinates the various agents, manages memory of past deals,
    and provides a high-level interface to run the system.
    """

    DB = "products_vectorstore"
    MEMORY_FILENAME = "memory.json"

    def __init__(self):
        """
        Initializes the DealAgentFramework.
        This includes setting up logging, loading environment variables,
        connecting to the ChromaDB vector store, and reading past opportunities from memory.
        """
        init_logging()
        load_dotenv()
        client = chromadb.PersistentClient(path=self.DB)
        self.memory = self.read_memory()
        self.collection = client.get_or_create_collection('products')
        self.planner = None

    def init_agents_as_needed(self):
        """
        Initializes the PlanningAgent if it hasn't been already.
        This is done lazily to avoid unnecessary initialization.
        """
        if not self.planner:
            self.log("Initializing Agent Framework")
            self.planner = PlanningAgent(self.collection)
            self.log("Agent Framework is ready")
        
    def read_memory(self) -> List[Opportunity]:
        """
        Reads the list of previously found opportunities from a JSON file.

        Returns:
            A list of Opportunity objects.
        """
        if os.path.exists(self.MEMORY_FILENAME):
            with open(self.MEMORY_FILENAME, "r") as file:
                data = json.load(file)
            opportunities = [Opportunity(**item) for item in data]
            return opportunities
        return []

    def write_memory(self) -> None:
        """
        Writes the current list of opportunities to a JSON file.
        """
        data = [opportunity.dict() for opportunity in self.memory]
        with open(self.MEMORY_FILENAME, "w") as file:
            json.dump(data, file, indent=2)

    def log(self, message: str):
        """
        Logs a message with a special formatting for the framework.
        
        Args:
            message: The message to be logged.
        """
        text = BG_BLUE + WHITE + "[Agent Framework] " + message + RESET
        logging.info(text)

    def run(self) -> List[Opportunity]:
        """
        Runs the main workflow of the agent framework.
        It initializes the agents, runs the planner to find new opportunities,
        and updates the memory with any new opportunities found.

        Returns:
            The updated list of all opportunities found so far.
        """
        self.init_agents_as_needed()
        logging.info("Kicking off Planning Agent")
        result = self.planner.plan(memory=self.memory)
        logging.info(f"Planning Agent has completed and returned: {result}")
        if result:
            self.memory.append(result)
            self.write_memory()
        return self.memory

    @classmethod
    def get_plot_data(cls, max_datapoints=10000):
        """
        Retrieves data for plotting the product embeddings.
        It fetches embeddings from the ChromaDB, reduces their dimensionality using t-SNE,
        and assigns colors based on their categories.

        Args:
            max_datapoints: The maximum number of data points to retrieve for the plot.

        Returns:
            A tuple containing the product documents, the reduced vectors, and the corresponding colors.
        """
        client = chromadb.PersistentClient(path=cls.DB)
        collection = client.get_or_create_collection('products')
        result = collection.get(include=['embeddings', 'documents', 'metadatas'], limit=max_datapoints)
        vectors = np.array(result['embeddings'])
        documents = result['documents']
        categories = [metadata['category'] for metadata in result['metadatas']]
        colors = [COLORS[CATEGORIES.index(c)] for c in categories]
        tsne = TSNE(n_components=3, random_state=42, n_jobs=-1)
        reduced_vectors = tsne.fit_transform(vectors)
        return documents, reduced_vectors, colors


if __name__=="__main__":
    DealAgentFramework().run()
    