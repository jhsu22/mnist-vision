"""
Main UI application
"""

import customtkinter as ctk

from config import App, initialize_fonts
from ui.container_frame import BaseFrame
from ui.demo_frame import DemoFrame
from ui.settings_frame import SettingsFrame
from ui.test_frame import TestFrame
from ui.train_frame import TrainFrame


class MnistPlayground(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Initialize fonts
        initialize_fonts()

        # Window setup
        self.title(App.TITLE)
        self.geometry(App.GEOMETRY)

        self.setup_ui()

    def setup_ui(self):
        self.container = BaseFrame(self)
        self.container.pack(fill="both", expand=True)

        content_frame = self.container.content_frame

        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=1)

        self.train_ui = TrainFrame(content_frame)
        self.train_ui.grid(row=0, column=0, sticky="nsew")

        self.test_ui = TestFrame(content_frame)
        self.test_ui.grid(row=0, column=0, sticky="nsew")

        self.demo_ui = DemoFrame(content_frame)
        self.demo_ui.grid(row=0, column=0, sticky="nsew")

        self.settings_ui = SettingsFrame(content_frame)
        self.settings_ui.grid(row=0, column=0, sticky="nsew")

        self.raise_frame(self.train_ui)

    def raise_frame(self, frame_to_raise):
        frame_to_raise.tkraise()


if __name__ == "__main__":
    app = MnistPlayground()
    app.mainloop()
