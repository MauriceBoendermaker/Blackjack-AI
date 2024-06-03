from lib.interfaces.gui import GraphicalUserInterface

def init_gui():
    gui = GraphicalUserInterface()
    gui.mainloop()
    gui.test()

def main():
    init_gui()

if __name__ == "__main__":
    main()
