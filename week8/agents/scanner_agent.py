import os
import json
from typing import Optional, List
from openai import OpenAI
from agents.deals import ScrapedDeal, DealSelection
from agents.agent import Agent


class ScannerAgent(Agent):
    """
    The ScannerAgent is responsible for finding and summarizing the best deals from a list of scraped deals.
    It uses a large language model (LLM), specifically GPT-4o-mini, to identify the 5 deals with the most
    detailed descriptions and clearest prices. It then returns these deals in a structured JSON format.
    """

    MODEL = "gpt-4o-mini"

    # The system prompt provides instructions to the LLM on how to behave.
    # It specifies the task, the desired output format (JSON), and important constraints.
    SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list, by selecting deals that have the most detailed, high quality description and the most clear price.
    Respond strictly in JSON with no explanation, using this format. You should provide the price as a number derived from the description. If the price of a deal isn't clear, do not include that deal in your response.
    Most important is that you respond with the 5 deals that have the most detailed product description with price. It's not important to mention the terms of the deal; most important is a thorough description of the product.
    Be careful with products that are described as \"$XXX off\" or \"reduced by $XXX\" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price. 
    
    {\"deals\": [
        {
            \"product_description\": \"Your clearly expressed summary of the product in 4-5 sentences. Details of the item are much more important than why it's a good deal. Avoid mentioning discounts and coupons; focus on the item itself. There should be a paragpraph of text for each item you choose.\",
            \"price\": 99.99,
            \"url\": \"the url as provided\"
        },
        ...
    ]}"""
    
    # The prefix for the user prompt, providing context and instructions.
    USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list, selecting those which have the most detailed, high quality product description and a clear price that is greater than 0.
    Respond strictly in JSON, and only JSON. You should rephrase the description to be a summary of the product itself, not the terms of the deal.
    Remember to respond with a paragraph of text in the product_description field for each of the 5 items that you select.
    Be careful with products that are described as \"$XXX off\" or \"reduced by $XXX\" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price. 
    
    Deals:
    
    """

    # The suffix for the user prompt, reinforcing the output format constraints.
    USER_PROMPT_SUFFIX = "\n\nStrictly respond in JSON and include exactly 5 deals, no more."

    name = "Scanner Agent"
    color = Agent.CYAN

    def __init__(self):
        """
        Initializes the ScannerAgent by setting up the OpenAI client.
        """
        self.log("Scanner Agent is initializing")
        self.openai = OpenAI()
        self.log("Scanner Agent is ready")

    def fetch_deals(self, memory) -> List[ScrapedDeal]:
        """
        Fetches new deals from RSS feeds and filters out any deals that are already in memory.

        Args:
            memory: A list of Opportunity objects representing deals that have been seen before.

        Returns:
            A list of new, unseen ScrapedDeal objects.
        """
        self.log("Scanner Agent is about to fetch deals from RSS feed")
        # Create a list of URLs from the memory to check for duplicates.
        urls = [opp.deal.url for opp in memory]
        # Fetch all deals from the RSS feeds.
        scraped = ScrapedDeal.fetch()
        # Filter out deals that have already been seen.
        result = [scrape for scrape in scraped if scrape.url not in urls]
        self.log(f"Scanner Agent received {len(result)} deals not already scraped")
        return result

    def make_user_prompt(self, scraped: List[ScrapedDeal]) -> str:
        """
        Creates the full user prompt to be sent to the LLM, including the prefix,
        the list of scraped deals, and the suffix.

        Args:
            scraped: A list of ScrapedDeal objects.

        Returns:
            The complete user prompt string.
        """
        user_prompt = self.USER_PROMPT_PREFIX
        user_prompt += '\n\n'.join([scrape.describe() for scrape in scraped])
        user_prompt += self.USER_PROMPT_SUFFIX
        return user_prompt

    def scan(self, memory: List[str]=[]) -> Optional[DealSelection]:
        """
        Scans for new deals and uses the LLM to select and summarize the best ones.
        It uses the OpenAI API's structured output feature to ensure the response is in the correct JSON format.

        Args:
            memory: A list of URLs of deals that have been seen before.

        Returns:
            A DealSelection object containing the selected deals, or None if no new deals are found.
        """
        scraped = self.fetch_deals(memory)
        if scraped:
            user_prompt = self.make_user_prompt(scraped)
            self.log("Scanner Agent is calling OpenAI using Structured Output")
            # Call the OpenAI API with the system and user prompts, and specify the desired response format.
            result = self.openai.beta.chat.completions.parse(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
              ],
                response_format=DealSelection
            )
            # Extract the parsed JSON content from the response.
            result = result.choices[0].message.parsed
            # Filter out any deals with a price of 0 or less.
            result.deals = [deal for deal in result.deals if deal.price>0]
            self.log(f"Scanner Agent received {len(result.deals)} selected deals with price>0 from OpenAI")
            return result
        return None
                
