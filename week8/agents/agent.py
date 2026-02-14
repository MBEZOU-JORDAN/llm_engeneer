import logging

class Agent:
    """
    An abstract superclass for Agents.
    It provides a standardized way for agents to log messages with a unique identifier,
    including a name and a color for easy identification in logs.
    """

    # ANSI escape codes for foreground colors
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # ANSI escape code for background color
    BG_BLACK = '\033[40m'
    
    # ANSI escape code to reset text formatting to default
    RESET = '\033[0m'

    # The name of the agent, to be overridden by subclasses.
    name: str = ""
    # The color associated with the agent for logging, defaults to white.
    color: str = '\033[37m'

    def log(self, message):
        """
        Logs a message with the agent's name and color.
        The message is logged at the INFO level.
        
        Args:
            message: The message to be logged.
        """
        # Combine background and foreground color codes for the log message.
        color_code = self.BG_BLACK + self.color
        # Format the message to include the agent's name.
        message = f"[{self.name}] {message}"
        # Log the colored message and then reset the color.
        logging.info(color_code + message + self.RESET)