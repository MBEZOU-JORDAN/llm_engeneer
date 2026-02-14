import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

from agents.agent import Agent
from agents.specialist_agent import SpecialistAgent
from agents.frontier_agent import FrontierAgent
from agents.random_forest_agent import RandomForestAgent

class EnsembleAgent(Agent):
    """
    The EnsembleAgent combines predictions from multiple other agents 
    (Specialist, Frontier, and RandomForest) to produce a final, weighted price estimate.
    It uses a pre-trained linear regression model to determine the weights for each agent's prediction.
    """

    name = "Ensemble Agent"
    color = Agent.YELLOW
    
    def __init__(self, collection):
        """
        Initializes the EnsembleAgent.
        This involves creating instances of the specialist, frontier, and random forest agents,
        and loading the pre-trained ensemble model from a file.

        Args:
            collection: A collection of items, likely used by the FrontierAgent.
        """
        self.log("Initializing Ensemble Agent")
        # Instantiate the individual agents that form the ensemble.
        self.specialist = SpecialistAgent()
        self.frontier = FrontierAgent(collection)
        self.random_forest = RandomForestAgent()
        # Load the pre-trained linear regression model for ensembling.
        self.model = joblib.load('ensemble_model.pkl')
        self.log("Ensemble Agent is ready")

    def price(self, description: str) -> float:
        """
        Executes the ensemble model to price a product.
        It gets price estimates from each of the sub-agents and then uses the
        linear regression model to combine them into a single, final price estimate.

        Args:
            description: The description of the product to be priced.

        Returns:
            The estimated price of the product.
        """
        self.log("Running Ensemble Agent - collaborating with specialist, frontier and random forest agents")
        
        # Get price estimates from each of the individual agents.
        specialist_price = self.specialist.price(description)
        frontier_price = self.frontier.price(description)
        random_forest_price = self.random_forest.price(description)
        
        # Create a DataFrame with the predictions from the individual agents,
        # as well as the min and max of their predictions. This is the input for the ensemble model.
        X = pd.DataFrame({
            'Specialist': [specialist_price],
            'Frontier': [frontier_price],
            'RandomForest': [random_forest_price],
            'Min': [min(specialist_price, frontier_price, random_forest_price)],
            'Max': [max(specialist_price, frontier_price, random_forest_price)],
        })
        
        # Use the pre-trained model to predict the final price based on the inputs.
        predicted_price = self.model.predict(X)[0]
        
        self.log(f"Ensemble Agent complete - returning ${predicted_price:.2f}")
        return predicted_price
