import os

import lib.common.constants as constants
from lib.interfaces.gui import GraphicalUserInterface


# TODO: Add logging to everything instead of ugly prints


# print("Current working directory:", os.getcwd())


def init_gui():
    gui = GraphicalUserInterface()
    gui.mainloop()


def main():
    print(f"Starting {constants.TITLE}")
    init_gui()


if __name__ == "__main__":
    main()
