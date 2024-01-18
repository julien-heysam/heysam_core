import logging
from rich.logging import RichHandler

# Configure logging to use RichHandler
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, tracebacks_show_locals=True)]
)

# Now import other modules that use logging
from sentence_transformers import SentenceTransformer
import src  # Assuming 'src' is your custom module

# Rest of your application code
if __name__ == '__main__':
    logging.info("This is an info message from the main module.")
