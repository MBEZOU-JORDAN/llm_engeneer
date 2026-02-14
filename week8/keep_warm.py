import time
import modal
from datetime import datetime

# This script is designed to keep a remote Modal service "warm" by periodically sending requests to it.
# This prevents the service from scaling down to zero and ensures that it's always ready to handle requests.

# Look up the "Pricer" class from the "pricer-service" app on Modal.
Pricer = modal.Cls.lookup("pricer-service", "Pricer")
# Create an instance of the remote Pricer class.
pricer = Pricer()

# Enter an infinite loop to periodically send requests.
while True:
    # Make a remote call to the 'wake_up' method of the Pricer object on Modal.
    reply = pricer.wake_up.remote()
    # Print the current time and the reply from the service.
    print(f"{datetime.now()}: {reply}")
    # Wait for 30 seconds before sending the next request.
    time.sleep(30)
