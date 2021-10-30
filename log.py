import logging
import sys

stdout_handler = [logging.StreamHandler(sys.stdout)]

logging.basicConfig(
    format='%(name)s - %(message)s',
    level=logging.INFO,
    handlers=stdout_handler
)

log = logging.getLogger(__name__)