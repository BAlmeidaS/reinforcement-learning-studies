import os
import json
import logging.config


def setup_logging(log_name="debug", path="logging.json"):
    if os.path.exists(path):
        with open(path, "rt") as f:
            config = json.load(f)
        config["handlers"]["info_file_handler"]["filename"] = log_name + ".txt"
        logging.config.dictConfig(config)
    else:
        raise FileNotFoundError(f"{path} not found")
