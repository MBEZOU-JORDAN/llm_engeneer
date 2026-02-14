import modal
from modal import App, Image

# Setup

# Create a Modal App named "hello". This serves as a container for our functions.
app = modal.App("hello")
# Define a container image using Debian Slim as the base and install the 'requests' library.
image = Image.debian_slim().pip_install("requests")

# Hello!

@app.function(image=image)
def hello() -> str:
    """
    A simple Modal function that makes a request to an IP geolocation API
    and returns a greeting with the location information.
    
    Returns:
        A string containing a greeting with the city, region, and country.
    """
    import requests
    
    # Make a GET request to the ipinfo.io API to get geolocation data.
    response = requests.get('https://ipinfo.io/json')
    # Parse the JSON response.
    data = response.json()
    # Extract the city, region, and country from the response data.
    city, region, country = data['city'], data['region'], data['country']
    # Return a formatted greeting string.
    return f"Hello from {city}, {region}, {country}!!"
