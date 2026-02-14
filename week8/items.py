from typing import Optional
from transformers import AutoTokenizer
import re

# Constants for the item processing.
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B"
MIN_TOKENS = 150
MAX_TOKENS = 160
MIN_CHARS = 300
CEILING_CHARS = MAX_TOKENS * 7

class Item:
    """
    Represents a cleaned and curated data point for a product, including its price.
    This class is responsible for parsing raw product data, cleaning it,
    and formatting it into a prompt suitable for training a language model.
    """
    
    # Initialize the tokenizer from the pre-trained model.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    # Prefix for the price in the prompt.
    PREFIX = "Price is $"
    # The question to be included in the prompt.
    QUESTION = "How much does this cost to the nearest dollar?"
    # A list of common, irrelevant phrases to be removed from the product details.
    REMOVALS = ['"Batteries Included?": "No"', '"Batteries Included?": "Yes"', '"Batteries Required?": "No"', '"Batteries Required?": "Yes"', "By Manufacturer", "Item", "Date First", "Package", ":", "Number of", "Best Sellers", "Number", "Product "]

    title: str
    price: float
    category: str
    token_count: int = 0
    details: Optional[str]
    prompt: Optional[str] = None
    # A flag to indicate whether this item should be included in the dataset.
    include = False

    def __init__(self, data, price):
        """
        Initializes an Item instance.

        Args:
            data: A dictionary containing the raw product data (title, description, features, details).
            price: The price of the product.
        """
        self.title = data['title']
        self.price = price
        self.parse(data)

    def scrub_details(self):
        """
        Cleans up the product details string by removing common, uninformative text.
        """
        details = self.details
        for remove in self.REMOVALS:
            details = details.replace(remove, "")
        return details

    def scrub(self, stuff):
        """
        Cleans up a given text string by removing unnecessary characters, whitespace,
        and likely irrelevant product numbers (words of 7+ characters containing numbers).
        
        Args:
            stuff: The text string to be cleaned.
            
        Returns:
            The cleaned text string.
        """
        stuff = re.sub(r'[:\[\]"{}【】\s]+', ' ', stuff).strip()
        stuff = stuff.replace(" ,", ",").replace(",,,",",").replace(",,",",")
        words = stuff.split(' ')
        # Filter out long words with numbers, which are likely product codes.
        select = [word for word in words if len(word)<7 or not any(char.isdigit() for char in word)]
        return " ".join(select)
    
    def parse(self, data):
        """
        Parses the raw product data, cleans it, and determines if it's suitable for inclusion
        in the dataset based on its length in characters and tokens.
        
        Args:
            data: The raw product data dictionary.
        """
        contents = '\n'.join(data['description'])
        if contents:
            contents += '\n'
        features = '\n'.join(data['features'])
        if features:
            contents += features + '\n'
        self.details = data['details']
        if self.details:
            contents += self.scrub_details() + '\n'
        
        # Only proceed if the content is long enough.
        if len(contents) > MIN_CHARS:
            # Truncate the content to a maximum character length.
            contents = contents[:CEILING_CHARS]
            text = f"{self.scrub(self.title)}\n{self.scrub(contents)}"
            # Tokenize the text to check its length.
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Only proceed if the token count is high enough.
            if len(tokens) > MIN_TOKENS:
                # Truncate the tokens to the maximum allowed length.
                tokens = tokens[:MAX_TOKENS]
                text = self.tokenizer.decode(tokens)
                # Create the prompt and mark the item for inclusion.
                self.make_prompt(text)
                self.include = True

    def make_prompt(self, text):
        """
        Creates the final training prompt for the item.
        The prompt includes the question, the cleaned product text, and the price.
        
        Args:
            text: The cleaned and truncated product text.
        """
        self.prompt = f"{self.QUESTION}\n\n{text}\n\n"
        self.prompt += f"{self.PREFIX}{str(round(self.price))}.00"
        # Calculate the final token count of the prompt.
        self.token_count = len(self.tokenizer.encode(self.prompt, add_special_tokens=False))

    def test_prompt(self):
        """
        Returns a version of the prompt suitable for testing,
        with the actual price removed.
        """
        return self.prompt.split(self.PREFIX)[0] + self.PREFIX

    def __repr__(self):
        """
        Returns a string representation of the Item.
        """
        return f"<{self.title} = ${self.price}>"

        

    
    