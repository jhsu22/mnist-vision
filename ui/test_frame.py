import customtkinter as ctk
import threading

from CTkMessagebox import CTkMessagebox

from config import LAYER_PARAMS, App, Paths
from core.model import BaseModel, get_layer_output_shapes
import os

import torch
import torch.nn as nn


class TestFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

        self.grid_columnconfigure(0, weight=1)

        # Track current model
        self.current_model = None           # Actual PyTorch model
        self.current_model_name = None      # Name of the current model

        self.setup_model_selection()

    def _get_saved_models(self):
        """Get a list of saved models"""
        if not os.path.exists(Paths.SAVED_MODELS_DIR):
            return []

        models = []

        for file in os.listdir(Paths.SAVED_MODELS_DIR):
            if file.endswith(".pth"):
                models.append(file[:-4])

        return sorted(models)

    def _refresh_model_list(self):
        """Refresh model selection dropdown"""
        saved_models = self._get_saved_models()

        if saved_models:
            all_models = saved_models
        else:
            all_models = []

        # Update model selection dropdown
        self.model_selection.configure(values=all_models)

        if self.current_model_name and self.current_model_name in all_models:
            self.model_selection.set(self.current_model_name)
        else:
            self.model_selection.set(all_models[0])

    def setup_model_selection(self):
        self.model_selection_frame = ctk.CTkFrame(self)
        self.model_selection_frame.grid(row=0, column=0, padx=10, pady=10)

        self.model_selection_frame.grid_columnconfigure(0, weight=1)
        self.model_selection_frame.grid_columnconfigure(1, weight=0)

        self.model_selection_label = ctk.CTkLabel(
            self.model_selection_frame, text="Select Model"
        )
        self.model_selection_label.grid(row=0, column=0, padx=5, sticky="w")

        self.model_selection = ctk.CTkComboBox(
            self.model_selection_frame, values=App.MODEL_LIST
        )
        self.model_selection.grid(row=0, column=1, padx=5, sticky="w")
