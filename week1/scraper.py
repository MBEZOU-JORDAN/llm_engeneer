from bs4 import BeautifulSoup
from urllib.parse import urljoin
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

from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin # <--- 1. On importe cet outil

# ... (le reste du code ne change pas)

def fetch_website_links(url):
    response = requests.get(url, headers=header)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    clean_links = []
    
    # 2. On parcourt toutes les balises <a>
    for link in soup.find_all("a"):
        href = link.get("href")
        
        if href: # Si le lien n'est pas vide
            # 3. C'est ici que la magie opère : on recolle le morceau au domaine principal
            full_url = urljoin(url, href)
            clean_links.append(full_url)
            
    # 4. On utilise set() pour supprimer les doublons (optionnel mais propre)
    return list(set(clean_links))