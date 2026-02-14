import modal
from agents.agent import Agent


class SpecialistAgent(Agent):
    """
    The SpecialistAgent is a client for a fine-tuned Language Model (LLM) that is running remotely on Modal.
    This agent is specialized in pricing items based on their descriptions, leveraging the fine-tuned model's expertise.
    """

    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self):
        """
        Initializes the SpecialistAgent.
        This involves looking up and connecting to the remote "pricer-service" on Modal.
        """
        self.log("Specialist Agent is initializing - connecting to modal")
        # Look up the "Pricer" class from the "pricer-service" app on Modal.
        Pricer = modal.Cls.lookup("pricer-service", "Pricer")
        # Create an instance of the remote Pricer class.
        self.pricer = Pricer()
        self.log("Specialist Agent is ready")
        
    def price(self, description: str) -> float:
        """
        Estimates the price of a product by making a remote call to the fine-tuned model on Modal.

        Args:
            description: The description of the product to be priced.

        Returns:
            The estimated price of the product as a float.
        """
        self.log("Specialist Agent is calling remote fine-tuned model")
        # Make a remote call to the 'price' method of the Pricer object on Modal.
        result = self.pricer.price.remote(description)
        self.log(f"Specialist Agent completed - predicting ${result:.2f}")
        return result
