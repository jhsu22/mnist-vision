import customtkinter as ctk
from config import Paths

from ui.app import MnistPlayground


def main():
    # Global application configuration
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme(Paths.THEME_PATH)

    # Create the application window
    app = MnistPlayground()

    # Run the application
    app.mainloop()


if __name__ == "__main__":
    main()
