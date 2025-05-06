import os
import json

SETTINGS_FILE = os.path.expanduser("~/.ptycho_gui_settings.json")


def load_settings(self):
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)
        self.previous_dir = settings.get("previous_dir", os.getcwd())
        self.last_file_count = settings.get("last_file_count", 1)
    else:
        self.previous_dir = os.getcwd()
        self.last_file_count = 1


def save_settings(self):
    settings = {
        "previous_dir": self.previous_dir,
        "last_file_count": self.last_file_count,
    }
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f)
