from typing import Optional, List
from agents.agent import Agent
from agents.deals import ScrapedDeal, DealSelection, Deal, Opportunity
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent


class PlanningAgent(Agent):
    """
    The PlanningAgent orchestrates the entire deal-finding workflow.
    It coordinates the actions of the ScannerAgent, EnsembleAgent, and MessagingAgent
    to find, evaluate, and report on deals.
    """

    name = "Planning Agent"
    color = Agent.GREEN
    # The minimum discount percentage required to consider a deal an opportunity worth alerting.
    DEAL_THRESHOLD = 50

    def __init__(self, collection):
        """
        Initializes the PlanningAgent.
        This involves creating instances of the ScannerAgent, EnsembleAgent, and MessagingAgent.

        Args:
            collection: The ChromaDB collection, which will be passed to the EnsembleAgent.
        """
        self.log("Planning Agent is initializing")
        # Instantiate the agents that this planner will coordinate.
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        self.messenger = MessagingAgent()
        self.log("Planning Agent is ready")

    def run(self, deal: Deal) -> Opportunity:
        """
        Processes a single deal to determine if it's an opportunity.
        It uses the EnsembleAgent to estimate the price of the deal's product
        and then calculates the discount.

        Args:
            deal: The deal to be evaluated.

        Returns:
            An Opportunity object containing the deal, its estimated price, and the discount.
        """
        self.log("Planning Agent is pricing up a potential deal")
        # Get a price estimate for the product from the ensemble agent.
        estimate = self.ensemble.price(deal.product_description)
        # Calculate the discount.
        discount = estimate - deal.price
        self.log(f"Planning Agent has processed a deal with discount ${discount:.2f}")
        return Opportunity(deal=deal, estimate=estimate, discount=discount)

    def plan(self, memory: List[str] = []) -> Optional[Opportunity]:
        """
        Executes the full deal-finding and notification workflow.
        1. Scans for new deals using the ScannerAgent.
        2. Evaluates the found deals using the `run` method.
        3. Identifies the best deal based on the discount.
        4. If the best deal's discount exceeds the threshold, it sends an alert using the MessagingAgent.

        Args:
            memory: A list of URLs of deals that have been seen before, to avoid duplicates.

        Returns:
            The best Opportunity found if its discount is above the threshold, otherwise None.
        """
        self.log("Planning Agent is kicking off a run")
        # Scan for new deals, avoiding those already in memory.
        selection = self.scanner.scan(memory=memory)
        if selection:
            # Process up to the first 5 deals found.
            opportunities = [self.run(deal) for deal in selection.deals[:5]]
            # Sort the opportunities by discount in descending order.
            opportunities.sort(key=lambda opp: opp.discount, reverse=True)
            best = opportunities[0]
            self.log(f"Planning Agent has identified the best deal has discount ${best.discount:.2f}")
            # If the best deal's discount is above the threshold, send an alert.
            if best.discount > self.DEAL_THRESHOLD:
                self.messenger.alert(best)
            self.log("Planning Agent has completed a run")
            # Return the best opportunity if it's good enough, otherwise None.
            return best if best.discount > self.DEAL_THRESHOLD else None
        return None
