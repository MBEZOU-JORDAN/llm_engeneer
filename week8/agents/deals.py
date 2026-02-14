from pydantic import BaseModel
from typing import List, Dict, Self
from bs4 import BeautifulSoup
import re
import feedparser
from tqdm import tqdm
import requests
import time

# List of RSS feed URLs for various deal categories.
feeds = [
    "https://www.dealnews.com/c142/Electronics/?rss=1",
    "https://www.dealnews.com/c39/Computers/?rss=1",
    "https://www.dealnews.com/c238/Automotive/?rss=1",
    "https://www.dealnews.com/f1912/Smart-Home/?rss=1",
    "https://www.dealnews.com/c196/Home-Garden/?rss=1",
]

def extract(html_snippet: str) -> str:
    """
    Uses Beautiful Soup to parse an HTML snippet and extract the useful text content.
    It specifically looks for a 'div' with the class 'snippet summary'.
    
    Args:
        html_snippet: A string containing the HTML to be parsed.
        
    Returns:
        A cleaned-up string containing the extracted text.
    """
    soup = BeautifulSoup(html_snippet, 'html.parser')
    snippet_div = soup.find('div', class_='snippet summary')
    
    if snippet_div:
        # Extract text and clean it up.
        description = snippet_div.get_text(strip=True)
        description = BeautifulSoup(description, 'html.parser').get_text()
        description = re.sub('<[^<]+?>', '', description)
        result = description.strip()
    else:
        # If the specific div is not found, return the original snippet.
        result = html_snippet
    return result.replace('\n', ' ')

class ScrapedDeal:
    """
    A class to represent a deal retrieved from an RSS feed.
    It scrapes additional details from the deal's URL.
    """
    category: str
    title: str
    summary: str
    url: str
    details: str
    features: str

    def __init__(self, entry: Dict[str, str]):
        """
        Initializes a ScrapedDeal instance from an RSS feed entry.
        
        Args:
            entry: A dictionary-like object representing an entry from an RSS feed.
        """
        self.title = entry['title']
        self.summary = extract(entry['summary'])
        self.url = entry['links'][0]['href']
        
        # Fetch and parse the content of the deal's URL.
        stuff = requests.get(self.url).content
        soup = BeautifulSoup(stuff, 'html.parser')
        content_div = soup.find('div', class_='content-section')
        if content_div:
            content = content_div.get_text()
            content = content.replace('\nmore', '').replace('\n', ' ')
            # Split content into details and features if "Features" section exists.
            if "Features" in content:
                self.details, self.features = content.split("Features")
            else:
                self.details = content
                self.features = ""
        else:
            self.details = ""
            self.features = ""


    def __repr__(self):
        """
        Returns a string representation of the deal, which is its title.
        """
        return f"<{self.title}>"

    def describe(self):
        """
        Returns a longer, more descriptive string for the deal,
        suitable for use in calls to a language model.
        """
        return f"Title: {self.title}\nDetails: {self.details.strip()}\nFeatures: {self.features.strip()}\nURL: {self.url}"

    @classmethod
    def fetch(cls, show_progress : bool = False) -> List[Self]:
        """
        Retrieves all deals from the RSS feeds defined in the `feeds` list.
        
        Args:
            show_progress: If True, displays a progress bar using tqdm.
            
        Returns:
            A list of ScrapedDeal instances.
        """
        deals = []
        feed_iter = tqdm(feeds) if show_progress else feeds
        for feed_url in feed_iter:
            feed = feedparser.parse(feed_url)
            # Limit to the first 10 entries per feed.
            for entry in feed.entries[:10]:
                deals.append(cls(entry))
                time.sleep(0.5) # Be polite and don't hammer the server.
        return deals

class Deal(BaseModel):
    """
    A Pydantic model to represent a deal with a summary description, price, and URL.
    """
    product_description: str
    price: float
    url: str

class DealSelection(BaseModel):
    """
    A Pydantic model to represent a list of deals.
    """
    deals: List[Deal]

class Opportunity(BaseModel):
    """
    A Pydantic model to represent a potential opportunity.
    This is a deal where the estimated cost is higher than the offered price.
    """
    deal: Deal
    estimate: float
    discount: float
