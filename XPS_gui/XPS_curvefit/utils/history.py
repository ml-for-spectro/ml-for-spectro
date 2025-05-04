# history.py

import copy
import logging


class HistoryManager:
    def __init__(self):
        self.stack = []

    def push(self, state_dict):
        self.stack.append(copy.deepcopy(state_dict))
        logging.debug("Pushed to history. Stack size: %d", len(self.stack))

    def undo(self):
        if not self.stack:
            logging.info("Undo requested but history is empty.")
            return None
        state = self.stack.pop()
        logging.info("Popped from history. Stack size: %d", len(self.stack))
        return state

    def clear(self):
        self.stack.clear()

    def can_undo(self):
        return len(self.stack) > 1

    def __len__(self):  # ← Add this method
        return len(self.stack)
