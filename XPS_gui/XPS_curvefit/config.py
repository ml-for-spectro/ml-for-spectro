import configparser
import os

CONFIG_FILE = os.path.expanduser("~/.xps_tool_config")


def load_last_dir():
    config = configparser.ConfigParser()
    if os.path.exists(CONFIG_FILE):
        config.read(CONFIG_FILE)
        return config.get("Settings", "last_dir", fallback="")
    return ""


def save_last_dir(path):
    config = configparser.ConfigParser()
    config["Settings"] = {"last_dir": path}
    with open(CONFIG_FILE, "w") as f:
        config.write(f)
