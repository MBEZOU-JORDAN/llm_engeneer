import os
# from twilio.rest import Client
from agents.deals import Opportunity
import http.client
import urllib
from agents.agent import Agent

# Set these constants to True to enable the respective messaging services.
# Note: You will need to have the appropriate API keys and credentials set up
# as environment variables for these services to work.

# To use Twilio for SMS, uncomment the Twilio import and the client initialization line.
DO_TEXT = False 
# To use Pushover for push notifications.
DO_PUSH = True

class MessagingAgent(Agent):
    """
    The MessagingAgent is responsible for sending notifications about potential deals.
    It can be configured to send notifications via SMS (using Twilio) or
    push notifications (using Pushover).
    """

    name = "Messaging Agent"
    color = Agent.WHITE

    def __init__(self):
        """
        Initializes the MessagingAgent.
        It sets up the necessary clients and credentials for Twilio and/or Pushover,
        based on the `DO_TEXT` and `DO_PUSH` constants.
        """
        self.log(f"Messaging Agent is initializing")
        if DO_TEXT:
            # Load Twilio credentials from environment variables.
            account_sid = os.getenv('TWILIO_ACCOUNT_SID', 'your-sid-if-not-using-env')
            auth_token = os.getenv('TWILIO_AUTH_TOKEN', 'your-auth-if-not-using-env')
            self.me_from = os.getenv('TWILIO_FROM', 'your-phone-number-if-not-using-env')
            self.me_to = os.getenv('MY_PHONE_NUMBER', 'your-phone-number-if-not-using-env')
            # self.client = Client(account_sid, auth_token) # Uncomment to use Twilio
            self.log("Messaging Agent has initialized Twilio")
        if DO_PUSH:
            # Load Pushover credentials from environment variables.
            self.pushover_user = os.getenv('PUSHOVER_USER', 'your-pushover-user-if-not-using-env')
            self.pushover_token = os.getenv('PUSHOVER_TOKEN', 'your-pushover-token-if-not-using-env')
            self.log("Messaging Agent has initialized Pushover")

    def message(self, text):
        """
        Sends an SMS message using the Twilio API.
        
        Args:
            text: The content of the message to be sent.
        """
        self.log("Messaging Agent is sending a text message")
        # This line will cause an error if the Twilio client is not initialized.
        message = self.client.messages.create(
          from_=self.me_from,
          body=text,
          to=self.me_to
        )

    def push(self, text):
        """
        Sends a push notification using the Pushover API.
        
        Args:
            text: The content of the push notification.
        """
        self.log("Messaging Agent is sending a push notification")
        conn = http.client.HTTPSConnection("api.pushover.net:443")
        conn.request("POST", "/1/messages.json",
          urllib.parse.urlencode({
            "token": self.pushover_token,
            "user": self.pushover_user,
            "message": text,
            "sound": "cashregister" # Specify the notification sound.
          }), { "Content-type": "application/x-www-form-urlencoded" })
        conn.getresponse()

    def alert(self, opportunity: Opportunity):
        """
        Sends an alert about a given opportunity.
        The alert can be a text message, a push notification, or both,
        depending on the configuration.

        Args:
            opportunity: The Opportunity object containing details about the deal.
        """
        # Format the alert text with details from the opportunity.
        text = f"Deal Alert! Price=${opportunity.deal.price:.2f}, "
        text += f"Estimate=${opportunity.estimate:.2f}, "
        text += f"Discount=${opportunity.discount:.2f} : "
        text += opportunity.deal.product_description[:10]+'... '
        text += opportunity.deal.url
        
        # Send the alert via the configured channels.
        if DO_TEXT:
            self.message(text)
        if DO_PUSH:
            self.push(text)
        self.log("Messaging Agent has completed")
        
    
        