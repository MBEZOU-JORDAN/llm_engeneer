# This module provides utilities for colorizing log messages for display in an HTML context.
# It defines ANSI color codes and a function to convert them into HTML `<span>` tags with corresponding colors.

# ANSI escape codes for foreground colors
RED = '\033[31m'
GREEN = '\033[32m'
YELLOW = '\033[33m'
BLUE = '\033[34m'
MAGENTA = '\033[35m'
CYAN = '\033[36m'
WHITE = '\033[37m'

# ANSI escape codes for background colors
BG_BLACK = '\033[40m'
BG_BLUE = '\033[44m'

# ANSI escape code to reset text formatting to default
RESET = '\033[0m'

# A dictionary mapping ANSI color code combinations to their corresponding HTML hex color codes.
mapper = {
    BG_BLACK+RED: "#dd0000",
    BG_BLACK+GREEN: "#00dd00",
    BG_BLACK+YELLOW: "#dddd00",
    BG_BLACK+BLUE: "#0000ee",
    BG_BLACK+MAGENTA: "#aa00dd",
    BG_BLACK+CYAN: "#00dddd",
    BG_BLACK+WHITE: "#87CEEB",
    BG_BLUE+WHITE: "#ff7800"
}


def reformat(message):
    """
    Reformats a log message containing ANSI color codes into an HTML string
    with `<span>` tags for colorization.

    Args:
        message: The log message string with ANSI color codes.

    Returns:
        An HTML formatted string with colors.
    """
    # Replace each ANSI color code with its corresponding HTML `<span>` tag.
    for key, value in mapper.items():
        message = message.replace(key, f'<span style="color: {value}">')
    # Replace the ANSI reset code with a closing `</span>` tag.
    message = message.replace(RESET, '</span>')
    return message
    
    