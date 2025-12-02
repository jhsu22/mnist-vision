import customtkinter as ctk

from config import App


class BaseFrame(ctk.CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)

        # Variable to store selected UI window
        self.window_selected = 0

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)

        self.title_frame = ctk.CTkFrame(self, fg_color="#101010")
        self.title_frame.grid(row=0, column=0, sticky="nsew")

        self.title = ctk.CTkLabel(self.title_frame, text=App.TITLE, font=App.FONT_TITLE)
        self.title.grid(row=0, column=0, padx=10, ipady=5, pady=5)

        self.header_frame = ctk.CTkFrame(self, fg_color="#101010")
        self.header_frame.grid(row=1, column=0, sticky="nsew")

        self.train_button = ctk.CTkButton(
            self.header_frame,
            text="Edit Model",
            command=lambda: parent.raise_frame(parent.train_ui),
        )
        self.train_button.pack(side="left", padx=10)

        self.test_button = ctk.CTkButton(
            self.header_frame,
            text="Test Model",
            command=lambda: parent.raise_frame(parent.test_ui),
        )
        self.test_button.pack(side="left", padx=10)

        self.demo_button = ctk.CTkButton(
            self.header_frame,
            text="Demo Model",
            command=lambda: parent.raise_frame(parent.demo_ui),
        )
        self.demo_button.pack(side="left", padx=10)

        self.settings_button = ctk.CTkButton(
            self.header_frame,
            text="Settings",
            command=lambda: parent.raise_frame(parent.settings_ui),
        )
        self.settings_button.pack(side="right", padx=10)

        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.grid(row=2, column=0, sticky="nsew")
