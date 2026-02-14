from bs4 import BeautifulSoup
import requests

# Standard headers to fetch a website
header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def fetch_website_contents(url):
    response = requests.get(url, headers=header)
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.title.string if soup.title else 'No title found'
    for irrelevant in soup(['script', 'style']):
        irrelevant.decompose()
        
    text = soup.body.get_text(separator="\n", strip=True) 
    return title + "\n\n" + text