import logging

LEVEL_SHORT = {
    "DEBUG": "D",
    "INFO": "I",
    "WARNING": "W",
    "ERROR": "E",
    "CRITICAL": "C"
}

class ShortLevelFormatter(logging.Formatter):
    def format(self, record):
        record.levelshort = LEVEL_SHORT.get(record.levelname, "?")
        return super().format(record)

def prepare_logger(logger, fmt_string="[%(name)s:%(levelshort)s] %(message)s"):
    handler = logging.StreamHandler()
    formatter = ShortLevelFormatter(fmt_string)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

egraph_logger = logging.getLogger("graph") # for merges, adds, etc.
rewrite_logger = logging.getLogger("rewrite")

logging.basicConfig(level=logging.WARNING)

loggers = {
    'EGRAPH': egraph_logger,
    'REWRITE': rewrite_logger,
}

for l in loggers.values():
    prepare_logger(l)

def logger_off(label):
    loggers[label].setLevel(logging.CRITICAL)

def debug(label):
    loggers[label].setLevel(logging.DEBUG)

def info(label):
    loggers[label].setLevel(logging.INFO)

def silence_all():
    for logger in loggers:
        logger_off(logger)