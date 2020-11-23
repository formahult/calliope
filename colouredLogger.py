import logging

class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    cyan = "\033[1;36m"
    GREEN = "\033[1;32m"
    MAGENTA = "\033[1;35m"
    BLUE = "\033[1;34m"
    BLACK = "\033[1;30m"
    YELLOW = "\033[1;33m"
    RED = "\033[1;31m"
    BOLD_RED = "\033[1;41m"
    reset = "\033[1;0m"
    #head = "%(asctime)s - %(name)s - "
    head = "%(name)s - "
    mid = "%(levelname)s"
    tail = " - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: head + cyan + mid + reset + tail,
        logging.INFO: head + GREEN + mid + reset + tail,
        logging.WARNING: head + YELLOW + mid + reset + tail,
        logging.ERROR: head + RED + mid + reset + tail,
        logging.CRITICAL: head + BOLD_RED + mid + reset + tail,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)