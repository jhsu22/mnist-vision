import os
import threading

import customtkinter as ctk
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from CTkMessagebox import CTkMessagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from config import LAYER_PARAMS, App, Paths
from core.model import BaseModel, get_layer_output_shapes


class TrainFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

        self.grid_columnconfigure(0, weight=0)  # Layer list - fixed
        self.grid_columnconfigure(1, weight=0)  # Param panel - fixed
        self.grid_columnconfigure(2, weight=0)  # Spacing
        self.grid_columnconfigure(3, weight=1)  # Plots - expandable

        # Initialize location variables
        self.layer_row = 0
        self.current_row = 0

        # Initialize layer list and index
        self.layers = []  # List of layer dictionaries
        self.selected_layer_index = None  # Track what layer is selected

        # Intialize training status
        self.training_status = False
        self.current_epoch = 0
        self.current_loss = 0
        self.current_accuracy = 0

        # Track current model
        self.current_model = None  # Actual PyTorch model
        self.current_model_name = None  # Name of the current model
        self.last_saved_layers = None  # For resetting

        # Initialize plot tracking variables
        self.update_interval = 50
        self.loss_history = []
        self.accuracy_history = []
        self.step_history = []

        # Create models directory
        os.makedirs(Paths.SAVED_MODELS_DIR, exist_ok=True)

        # Setup model selection
        self.setup_model_selection()

        # Setup layer editor
        self.setup_layer_editor()

        # Create hyperparameter inputs
        self.setup_hyperparameters()

        # Create plot area
        self.setup_plot()

        # Create bottom buttons
        self.setup_bottom_buttons()

        # Refresh model list
        self._refresh_model_list()

    def _get_toplevel(self):
        return self.winfo_toplevel()

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
        elif all_models:
            self.model_selection.set(all_models[0])
        else:
            self.model_selection.set("No models found")

    def _new_model(self):
        if self.layers and self.current_model is not None:
            msg = CTkMessagebox(
                master=self,
                title="New Model",
                message="Are you sure you want to create a new model?\nThis will discard your current model.",
                icon="question",
                option_1="No",
                option_2="Yes",
            )
            if msg.get() != "Yes":
                return

        # Clear internal states
        self.layers = []
        self.selected_layer_index = None
        self.current_model = None
        self.current_model_name = None
        self.last_saved_layers = None

        # Clear UI
        self._refresh_layer_list()
        self._display_layer_params(None)

        # Clear hyperparameter entries
        self.learning_rate_entry.delete(0, "end")
        self.epochs_entry.delete(0, "end")
        self.batch_size_entry.delete(0, "end")
        self.optimizer_entry.set("Adam")

        # Update model selector
        self.model_selection.set("")

        # Clear plot history
        self.loss_history = []
        self.accuracy_history = []
        self.step_history = []
        self._update_plots()

    def _save_model(self):
        """Save current model to .pth file"""
        if not self.layers:
            _ = CTkMessagebox(
                master=self,
                title="Save Model",
                message="Your model has no layers. Add layers first.",
                icon="warning",
            )
            return

        # Get model name from dialog
        dialog = ctk.CTkInputDialog(text="Enter model name:", title="Save Model")
        model_name = dialog.get_input()

        if not model_name:
            return  # If user cancels dialog

        # Remove invalid characters
        model_name = "".join(c for c in model_name if c.isalnum() or c in "._- ")
        model_name = model_name.strip()

        if not model_name:
            _ = CTkMessagebox(
                master=self,
                title="Save Model",
                message="Invalid model name.",
                icon="warning",
            )
            return

        # Check if file exists
        file_path = os.path.join(Paths.SAVED_MODELS_DIR, f"{model_name}.pth")
        if os.path.exists(file_path):
            msg = CTkMessagebox(
                master=self,
                title="Overwrite Model",
                message=f"Model '{model_name}' already exists. Overwrite?",
                icon="question",
                option_1="No",
                option_2="Yes",
            )
            if msg.get() != "Yes":
                return

        # Prepare model for saving
        save_data = {
            "layer_configs": self.layers,
            "hyperparameters": {
                "learning_rate": self.learning_rate_entry.get() or "0.001",
                "epochs": self.epochs_entry.get() or "10",
                "batch_size": self.batch_size_entry.get() or "32",
                "optimizer": self.optimizer_entry.get() or "Adam",
            },
            "loss_history": self.loss_history or [],
            "accuracy_history": self.accuracy_history or [],
            "step_history": self.step_history or [],
        }

        if self.current_model is not None:
            save_data["model_state_dict"] = self.current_model.state_dict()

        # Save model
        torch.save(save_data, file_path)
        self.current_model_name = model_name
        self.last_saved_layers = [dict(layer) for layer in self.layers]

        _ = CTkMessagebox(
            master=self,
            title="Save Model",
            message=f"Model saved as '{model_name}.pth'",
            icon="check",
        )
        self._refresh_model_list()

    def _load_model(self):
        """Load a model from a .pth file"""
        model_name = self.model_selection.get()

        if not model_name or model_name == "(No saved models)":
            _ = CTkMessagebox(
                master=self,
                title="Load Model",
                message="No model selected.",
                icon="warning",
            )
            return

        file_path = os.path.join(Paths.SAVED_MODELS_DIR, f"{model_name}.pth")

        if not os.path.exists(file_path):
            _ = CTkMessagebox(
                master=self,
                title="Load Model",
                message=f"Model '{model_name}' not found.",
                icon="cancel",
            )
            return

        # Load the model
        save_data = torch.load(file_path, weights_only=False)

        # Restore layer configuration
        self.layers = save_data.get("layer_configs", [])

        # Get hyperparameters
        hyperparams = save_data.get("hyperparameters", {})

        # Restore hyperparameters
        self.learning_rate_entry.delete(0, "end")
        self.learning_rate_entry.insert(0, hyperparams.get("learning_rate", "0.001"))

        self.epochs_entry.delete(0, "end")
        self.epochs_entry.insert(0, hyperparams.get("epochs", "10"))

        self.batch_size_entry.delete(0, "end")
        self.batch_size_entry.insert(0, hyperparams.get("batch_size", "32"))

        self.optimizer_entry.set(hyperparams.get("optimizer", "Adam"))

        # Rebuild model if the saved file has weights
        if "model_state_dict" in save_data and self.layers:
            self.current_model = BaseModel(self.layers)
            self.current_model.load_state_dict(save_data["model_state_dict"])
            self.current_model.eval()
        else:
            self.current_model = None

        self.current_model_name = model_name
        self.last_saved_layers = [dict(layer) for layer in self.layers]

        # Refresh UI
        self.selected_layer_index = None
        self._refresh_layer_list()
        self._display_layer_params(None)

        # Clear plot history
        self.loss_history = save_data.get("loss_history", [])
        self.accuracy_history = save_data.get("accuracy_history", [])
        self.step_history = save_data.get("step_history", [])
        self._update_plots()

        _ = CTkMessagebox(
            master=self,
            title="Load Model",
            message=f"Model '{model_name}' loaded successfully.",
            icon="check",
        )

    def _reset_model(self):
        if self.last_saved_layers is not None:
            msg = CTkMessagebox(
                master=self,
                title="Reset Model",
                message="Reset to last saved state?\n(Current changes will be lost)",
                icon="question",
                option_1="No",
                option_2="Yes",
            )
            if msg.get() == "Yes":
                self.layers = [dict(layer) for layer in self.last_saved_layers]
                self.selected_layer_index = None
                self._refresh_layer_list()
                self._display_layer_params(None)

        else:
            msg = CTkMessagebox(
                master=self,
                title="Reset Model",
                message="No saved state. Clear all layers?",
                icon="question",
                option_1="No",
                option_2="Yes",
            )
            if msg.get() == "Yes":
                self._new_model()

    def _add_layer(self):
        # Get layer type from user input
        layer_type = self.layer_selector.get()

        # Get parameter template for selected layer type
        param_template = LAYER_PARAMS[layer_type]

        # Extract default values
        default_params = {}
        for param_name, param_info in param_template.items():
            default_value = param_info["default"]
            # Skip "None" activation
            if param_name == "activation" and default_value == "None":
                continue
            default_params[param_name] = default_value

        # Create layer dictionary
        new_layer = {"type": layer_type, "params": default_params}

        # Add layer to list
        self.layers.append(new_layer)

        self._refresh_layer_list()

        # Auto-select the newly added layer
        self._select_layer(len(self.layers) - 1)

    def _refresh_layer_list(self):
        # Clear current layer list
        for item in self.layer_list_frame.winfo_children():
            item.destroy()

        # Get calculated shapes for all layers
        try:
            shapes = get_layer_output_shapes(self.layers)
        except Exception as e:
            shapes = []

        # Refresh layer list
        shape_idx = 0
        for index, layer in enumerate(self.layers):
            # Create a frame for the layer
            layer_item = ctk.CTkFrame(self.layer_list_frame)
            layer_item.grid(row=index, column=0, padx=5, pady=5, sticky="ew")
            layer_item.layer_index = index

            # Create label text with layer info
            layer_type = layer["type"]

            # Find the corresponding shape info
            shape_text = ""
            if shape_idx < len(shapes):
                # Skip auto-inserted flatten in shape display
                while (
                    shape_idx < len(shapes) and "auto" in shapes[shape_idx][0].lower()
                ):
                    shape_idx += 1

                if shape_idx < len(shapes):
                    layer_desc, shape = shapes[shape_idx]
                    if len(shape) == 2:
                        shape_text = f" → {shape[1]} features"
                    elif len(shape) == 4:
                        shape_text = f" → {shape[1]}ch × {shape[2]}×{shape[3]}"
                    shape_idx += 1

                    # Skip activation shapes
                    while shape_idx < len(shapes) and shapes[shape_idx][0].startswith(
                        "  └─"
                    ):
                        shape_idx += 1

            label_text = f"{index + 1}. {layer_type}{shape_text}"

            layer_label = ctk.CTkLabel(layer_item, text=label_text, anchor="w")
            layer_label.pack(side="left", padx=10, pady=5, fill="x", expand=True)

            layer_item.bind(
                "<Button-1>", lambda event, index=index: self._select_layer(index)
            )
            layer_label.bind(
                "<Button-1>", lambda event, index=index: self._select_layer(index)
            )

    def _select_layer(self, index):
        self.selected_layer_index = index

        # Loop through all layer items and update their colors
        for item in self.layer_list_frame.winfo_children():
            if hasattr(item, "layer_index"):
                if item.layer_index == index:
                    item.configure(fg_color="#2E2E2E")
                else:
                    item.configure(fg_color="#1A1A1A")

        self._display_layer_params(index)

    def _display_layer_params(self, index):
        # Clear widgets from param panel
        for widget in self.param_panel.winfo_children():
            widget.destroy()

        if index is None:
            # Show message when no layer is selected
            msg = ctk.CTkLabel(self.param_panel, text="Select a layer to edit")
            msg.pack(pady=20)

        else:  # Show selected layer parameters
            current_layer = self.layers[index]
            layer_type = current_layer["type"]

            param_template = LAYER_PARAMS[layer_type]

            # Show layer type header
            header = ctk.CTkLabel(
                self.param_panel,
                text=f"{layer_type} Layer",
                font=ctk.CTkFont(size=16, weight="bold"),
            )
            header.grid(
                row=0, column=0, columnspan=2, padx=5, pady=(10, 20), sticky="w"
            )

            if not param_template:
                # No parameters for this layer type
                msg = ctk.CTkLabel(self.param_panel, text="No configurable parameters")
                msg.grid(row=1, column=0, columnspan=2, padx=5, pady=10)
                return

            row_number = 1

            for param_name, param_info in param_template.items():
                # Get current value from the layer
                param_value = current_layer["params"].get(
                    param_name, param_info["default"]
                )

                # Create label
                param_label = ctk.CTkLabel(self.param_panel, text=param_info["label"])
                param_label.grid(row=row_number, column=0, padx=5, pady=10, sticky="w")

                if param_info["type"] == "int":
                    param_entry = ctk.CTkEntry(self.param_panel)
                    param_entry.insert(0, str(param_value))
                    param_entry.grid(
                        row=row_number, column=1, padx=5, pady=10, sticky="ew"
                    )

                    param_entry.bind(
                        "<Return>",
                        lambda event, pn=param_name: self._update_param(
                            pn, event.widget.get()
                        ),
                    )
                    # Also update on focus out
                    param_entry.bind(
                        "<FocusOut>",
                        lambda event, pn=param_name: self._update_param(
                            pn, event.widget.get()
                        ),
                    )

                elif param_info["type"] == "dropdown":
                    param_combo = ctk.CTkComboBox(
                        self.param_panel, values=param_info["options"]
                    )
                    param_combo.set(
                        param_value if param_value else param_info["options"][0]
                    )
                    param_combo.grid(
                        row=row_number, column=1, padx=5, pady=10, sticky="ew"
                    )

                    param_combo.configure(
                        command=lambda val, p=param_name: self._update_param(p, val)
                    )

                row_number += 1

    def _update_param(self, param_name, new_value):
        if self.selected_layer_index is None:
            return

        current_layer = self.layers[self.selected_layer_index]
        layer_type = current_layer["type"]
        param_info = LAYER_PARAMS[layer_type].get(param_name)

        if param_info is None:
            return

        if param_info["type"] == "int":
            try:
                new_value = int(new_value)
            except ValueError:
                return  # Invalid input, ignore

        # Handle "None" activation
        if param_name == "activation" and new_value == "None":
            if "activation" in current_layer["params"]:
                del current_layer["params"]["activation"]
        else:
            current_layer["params"][param_name] = new_value

        # Refresh layer list to update shape calculations
        self._refresh_layer_list()
        # Re-highlight selected layer
        self._select_layer(self.selected_layer_index)

    def setup_model_selection(self):
        self.model_selection_frame = ctk.CTkFrame(self)
        self.model_selection_frame.grid(
            row=self.current_row, column=0, columnspan=2, padx=10, pady=10, sticky="ew"
        )
        self.model_selection_frame.grid_columnconfigure(1, weight=1)

        self.model_selection_label = ctk.CTkLabel(
            self.model_selection_frame, text="Select Model"
        )
        self.model_selection_label.grid(row=0, column=0, padx=5, sticky="w")

        self.model_selection = ctk.CTkComboBox(
            self.model_selection_frame, values=App.MODEL_LIST
        )
        self.model_selection.grid(row=0, column=1, padx=5, sticky="ew")

        self.model_buttons_frame = ctk.CTkFrame(self.model_selection_frame)
        self.model_buttons_frame.grid(
            row=1, column=0, columnspan=2, padx=0, pady=10, sticky="w"
        )

        self.new_model_button = ctk.CTkButton(
            self.model_buttons_frame, text="New Model", command=self._new_model
        )
        self.new_model_button.grid(row=0, column=0, padx=5, pady=(5, 0), sticky="w")

        self.save_button = ctk.CTkButton(
            self.model_buttons_frame, text="Save Model", command=self._save_model
        )
        self.save_button.grid(row=0, column=1, padx=5, pady=(5, 0), sticky="w")

        self.reset_button = ctk.CTkButton(
            self.model_buttons_frame, text="Reset Model", command=self._reset_model
        )
        self.reset_button.grid(row=0, column=2, padx=5, pady=(5, 0), sticky="w")

        self.load_button = ctk.CTkButton(
            self.model_buttons_frame, text="Load Model", command=self._load_model
        )
        self.load_button.grid(row=0, column=3, padx=5, pady=(5, 0), sticky="w")

        self.current_row += 1

    def _delete_layer(self):
        if self.selected_layer_index is not None:
            del self.layers[self.selected_layer_index]
            self.selected_layer_index = None
            self._refresh_layer_list()
            self._display_layer_params(None)

    def _move_up(self):
        if self.selected_layer_index is not None and self.selected_layer_index > 0:
            (
                self.layers[self.selected_layer_index],
                self.layers[self.selected_layer_index - 1],
            ) = (
                self.layers[self.selected_layer_index - 1],
                self.layers[self.selected_layer_index],
            )
            self.selected_layer_index -= 1
            self._refresh_layer_list()
            self._select_layer(self.selected_layer_index)

    def _move_down(self):
        if (
            self.selected_layer_index is not None
            and self.selected_layer_index < len(self.layers) - 1
        ):
            (
                self.layers[self.selected_layer_index],
                self.layers[self.selected_layer_index + 1],
            ) = (
                self.layers[self.selected_layer_index + 1],
                self.layers[self.selected_layer_index],
            )
            self.selected_layer_index += 1
            self._refresh_layer_list()
            self._select_layer(self.selected_layer_index)

    def setup_layer_editor(self):
        self.layer_function_frame = ctk.CTkFrame(self)
        self.layer_function_frame.grid(
            row=self.current_row, column=0, columnspan=2, padx=10, pady=10, sticky="ew"
        )

        self.layer_function_frame.grid_columnconfigure(1, weight=1)

        self.layer_label = ctk.CTkLabel(self.layer_function_frame, text="Layer")
        self.layer_label.grid(row=0, column=0, padx=5, pady=10, sticky="w")

        self.layer_selector = ctk.CTkComboBox(
            self.layer_function_frame, values=App.LAYER_LIST
        )
        self.layer_selector.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        self.add_layer_button = ctk.CTkButton(
            self.layer_function_frame,
            text="Add Layer",
            command=self._add_layer,
        )
        self.add_layer_button.grid(row=0, column=2, padx=10, pady=10, sticky="w")

        self.current_row += 1

        self.layer_button_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.layer_button_frame.grid(
            row=self.current_row, column=0, columnspan=1, padx=10, pady=0, sticky="ew"
        )

        self.delete_button = ctk.CTkButton(
            self.layer_button_frame,
            text="Delete Layer",
            command=self._delete_layer,
        )
        self.delete_button.pack(side="right", padx=5)

        self.moveup_button = ctk.CTkButton(
            self.layer_button_frame,
            text="Move Up",
            command=self._move_up,
        )
        self.moveup_button.pack(side="right", padx=5)

        self.movedown_button = ctk.CTkButton(
            self.layer_button_frame,
            text="Move Down",
            command=self._move_down,
        )
        self.movedown_button.pack(side="right", padx=5)

        self.current_row += 1

        layer_panels_start_row = self.current_row
        self.layer_list_frame = ctk.CTkScrollableFrame(self, fg_color="#101010")
        self.layer_list_frame.grid(
            row=layer_panels_start_row,
            column=0,
            columnspan=1,
            rowspan=4,
            padx=15,
            pady=10,
            sticky="nsew",
        )
        self.layer_list_frame.grid_columnconfigure(0, weight=1)

        self.grid_rowconfigure(layer_panels_start_row, weight=1)
        self.current_row += 4

        self.param_panel = ctk.CTkFrame(self, fg_color="#101010")
        self.param_panel.grid(
            row=layer_panels_start_row,
            column=1,
            padx=10,
            pady=10,
            sticky="nsew",
        )
        self.param_panel.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(layer_panels_start_row, weight=1)

    def setup_hyperparameters(self):
        """Setup hyperparameter input boxes in a compact 2x2 grid"""
        self.hyperparameters_frame = ctk.CTkFrame(self)
        self.hyperparameters_frame.grid(
            row=self.current_row, column=0, columnspan=2, padx=15, pady=10, sticky="ew"
        )

        # Configure grid
        self.hyperparameters_frame.grid_columnconfigure(0, weight=1)
        self.hyperparameters_frame.grid_columnconfigure(1, weight=1)

        # Title
        title = ctk.CTkLabel(
            self.hyperparameters_frame,
            text="Hyperparameters",
            font=ctk.CTkFont(family="Albert Sans", size=14, weight="bold"),
        )
        title.grid(row=0, column=0, columnspan=2, padx=10, pady=(10, 15), sticky="w")

        self.learning_rate_label = ctk.CTkLabel(
            self.hyperparameters_frame, text="Learning Rate"
        )
        self.learning_rate_label.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        self.learning_rate_entry = ctk.CTkEntry(
            self.hyperparameters_frame, placeholder_text="0.001"
        )
        self.learning_rate_entry.grid(
            row=2, column=0, padx=10, pady=(0, 10), sticky="ew"
        )

        self.epochs_label = ctk.CTkLabel(self.hyperparameters_frame, text="Epochs")
        self.epochs_label.grid(row=1, column=1, padx=10, pady=5, sticky="w")

        self.epochs_entry = ctk.CTkEntry(
            self.hyperparameters_frame, placeholder_text="10"
        )
        self.epochs_entry.grid(row=2, column=1, padx=10, pady=(0, 10), sticky="ew")

        self.batch_size_label = ctk.CTkLabel(
            self.hyperparameters_frame, text="Batch Size"
        )
        self.batch_size_label.grid(row=3, column=0, padx=10, pady=5, sticky="w")

        self.batch_size_entry = ctk.CTkEntry(
            self.hyperparameters_frame, placeholder_text="32"
        )
        self.batch_size_entry.grid(row=4, column=0, padx=10, pady=(0, 10), sticky="ew")

        self.optimizer_label = ctk.CTkLabel(
            self.hyperparameters_frame, text="Optimizer"
        )
        self.optimizer_label.grid(row=3, column=1, padx=10, pady=5, sticky="w")

        self.optimizer_entry = ctk.CTkComboBox(
            self.hyperparameters_frame, values=App.OPTIMIZER_LIST
        )
        self.optimizer_entry.grid(row=4, column=1, padx=10, pady=(0, 10), sticky="ew")

        self.current_row += 1

    def _build_model(self):
        """Build the ML model based on the user inputs"""

        # Get hyperparameters from UI entries
        hyperparams = {
            "learning_rate": float(self.learning_rate_entry.get() or 0.001),
            "epochs": int(self.epochs_entry.get() or 10),
            "batch_size": int(self.batch_size_entry.get() or 32),
            "optimizer": self.optimizer_entry.get() or "Adam",
        }

        # Create model with auto-calculated sizes
        model = BaseModel(self.layers)

        # Create optimizer
        optimizer = model.get_optimizer(
            hyperparams["optimizer"], hyperparams["learning_rate"]
        )

        return model, optimizer, hyperparams

    def _update_display(self, widget, text):
        """Helper method to update textbox displays from any thread"""

        def update():
            widget.delete("1.0", "end")
            widget.insert("1.0", text)

        # Schedule the update on the main thread
        self.after(0, update)

    def _start_training(self):
        """Run training loop in a separate thread"""

        training_thread = threading.Thread(target=self._training_loop)
        training_thread.daemon = True
        training_thread.start()

    def _training_loop(self):
        """Training loop for ML model with per-batch updates"""
        try:
            # Build model
            model, optimizer, hyperparams = self._build_model()

            # Print model summary
            print("\n" + "=" * 60)
            model.summary()
            print("=" * 60 + "\n")

            device = torch.device("cpu")
            model.to(device)

            # Load data
            train_loader = self._get_data_loader(hyperparams["batch_size"])

            # Loss function
            criterion = nn.CrossEntropyLoss()

            self.training_status = True

            # Clear history before training
            self.loss_history = []
            self.accuracy_history = []
            self.step_history = []

            # Track running metrics within update window
            running_loss = 0.0
            running_correct = 0
            running_total = 0
            global_batch_count = 0  # Total batches across all epochs

            num_batches = len(train_loader)
            total_epochs = hyperparams["epochs"]

            # Training loop
            for epoch in range(total_epochs):
                if not self.training_status:
                    break

                model.train()
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_total = 0

                for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                    if not self.training_status:
                        break

                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                    # Forward pass
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Track metrics
                    batch_loss = loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    batch_size = batch_y.size(0)
                    batch_correct = (predicted == batch_y).sum().item()

                    # Accumulate for running average
                    running_loss += batch_loss
                    running_correct += batch_correct
                    running_total += batch_size

                    # Accumulate for epoch statistics
                    epoch_loss += batch_loss
                    epoch_correct += batch_correct
                    epoch_total += batch_size

                    global_batch_count += 1

                    # Update plot every N batches
                    if global_batch_count % self.update_interval == 0:
                        # Calculate averages over the update window
                        avg_loss = running_loss / self.update_interval
                        avg_accuracy = 100.0 * running_correct / running_total

                        # Calculate fractional epoch
                        fractional_epoch = epoch + (batch_idx + 1) / num_batches

                        # Store in history
                        self.step_history.append(fractional_epoch)
                        self.loss_history.append(avg_loss)
                        self.accuracy_history.append(avg_accuracy)

                        # Update plots on main thread
                        self.after(0, self._update_plots)

                        # Print progress
                        print(
                            f"Epoch {epoch + 1}/{total_epochs} "
                            f"[Batch {batch_idx + 1}/{num_batches}] - "
                            f"Loss: {avg_loss:.4f}, "
                            f"Accuracy: {avg_accuracy:.2f}%"
                        )

                        # Reset running metrics
                        running_loss = 0.0
                        running_correct = 0
                        running_total = 0

                # Print epoch summary
                epoch_avg_loss = epoch_loss / num_batches
                epoch_avg_accuracy = 100.0 * epoch_correct / epoch_total
                print(f"\n{'=' * 60}")
                print(f"EPOCH {epoch + 1}/{total_epochs} SUMMARY:")
                print(f"  Average Loss: {epoch_avg_loss:.4f}")
                print(f"  Average Accuracy: {epoch_avg_accuracy:.2f}%")
                print(f"{'=' * 60}\n")

            self.training_status = False

            # Store the trained model
            self.current_model = model
            self.current_model.eval()

            print("\n" + "=" * 60)
            print("TRAINING COMPLETE!")
            print("=" * 60 + "\n")

        except Exception as e:
            import traceback

            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self.training_status = False

    def _get_data_loader(self, batch_size):
        """Get training data"""
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from torchvision.datasets import ImageFolder

        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        train_dataset = ImageFolder(root="mnist_dataset/training", transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        return train_loader

    def _get_test_loader(self, batch_size):
        """Get the test data"""
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from torchvision.datasets import ImageFolder

        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        test_dataset = ImageFolder(root=Paths.TEST_DIR, transform=transform)

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return test_loader

    def _stop_training(self):
        self.training_status = False

    def setup_plot(self):
        """Setup matplotlib plots for training visualization"""

        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.grid(
            row=0, column=3, padx=10, pady=10, sticky="nsew", rowspan=self.current_row
        )

        # Make plot frame expand
        self.plot_frame.grid_rowconfigure(1, weight=1)
        self.plot_frame.grid_columnconfigure(0, weight=1)

        # Title label
        self.plot_label = ctk.CTkLabel(
            self.plot_frame,
            text="Training Progress",
            font=ctk.CTkFont(family="monospace", size=14, weight="bold"),
        )
        self.plot_label.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # Create matplotlib figure
        plt.style.use("dark_background")
        self.fig = Figure(figsize=(6, 8), dpi=100, facecolor="#1a1a1a")

        # Create subplots for loss and accuracy
        self.ax_loss = self.fig.add_subplot(211)
        self.ax_accuracy = self.fig.add_subplot(212)

        # Loss plot styling
        self.ax_loss.set_facecolor("#1a1a1a")
        self.ax_loss.set_xlabel(
            "Epoch", color="white", fontfamily="monospace", fontsize=10
        )
        self.ax_loss.set_ylabel(
            "Loss", color="white", fontfamily="monospace", fontsize=10
        )
        self.ax_loss.set_title(
            "Training Loss",
            color="white",
            fontfamily="monospace",
            fontsize=10,
            fontweight="bold",
            pad=10,
        )
        self.ax_loss.grid(True, alpha=0.2, linestyle="-", color="gray")
        self.ax_loss.tick_params(colors="white", labelsize=9)
        self.ax_loss.spines["top"].set_visible(False)
        self.ax_loss.spines["right"].set_visible(False)
        self.ax_loss.spines["left"].set_color("white")
        self.ax_loss.spines["bottom"].set_color("white")

        # Accuracy plot styling
        self.ax_accuracy.set_facecolor("#1a1a1a")
        self.ax_accuracy.set_xlabel(
            "Epoch", color="white", fontfamily="monospace", fontsize=10
        )
        self.ax_accuracy.set_ylabel(
            "Accuracy (%)", color="white", fontfamily="monospace", fontsize=10
        )
        self.ax_accuracy.set_title(
            "Training Accuracy",
            color="white",
            fontfamily="monospace",
            fontsize=10,
            fontweight="bold",
            pad=10,
        )
        self.ax_accuracy.grid(True, alpha=0.2, linestyle="-", color="gray")
        self.ax_accuracy.tick_params(colors="white", labelsize=9)
        self.ax_accuracy.spines["top"].set_visible(False)
        self.ax_accuracy.spines["right"].set_visible(False)
        self.ax_accuracy.spines["left"].set_color("white")
        self.ax_accuracy.spines["bottom"].set_color("white")

        self.fig.tight_layout(pad=3)

        # Create tkinter object
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(
            row=1, column=0, padx=10, pady=10, sticky="nsew"
        )

        # Initialize plots with red color
        (self.loss_line,) = self.ax_loss.plot([], [], color="#E74C3C", linewidth=2.5)
        (self.accuracy_line,) = self.ax_accuracy.plot(
            [], [], color="#E74C3C", linewidth=2.5
        )

    def _update_plots(self):
        """Update training plots with new data"""

        # Loss plot
        self.loss_line.set_data(self.step_history, self.loss_history)
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()

        # Accuracy plot
        self.accuracy_line.set_data(self.step_history, self.accuracy_history)
        self.ax_accuracy.relim()
        self.ax_accuracy.autoscale_view()

        # Redraw canvas
        self.fig.tight_layout(pad=3)
        self.canvas.draw()
        self.canvas.flush_events()

    def setup_bottom_buttons(self):
        self.bottom_button_frame = ctk.CTkFrame(self)
        self.bottom_button_frame.grid(
            row=self.current_row, column=0, columnspan=5, padx=10, pady=10, sticky="we"
        )

        self.bottom_button_frame.grid_columnconfigure(0, weight=1)

        self.current_row += 1

        self.train_button = ctk.CTkButton(
            self.bottom_button_frame, text="Train Model", command=self._start_training
        )
        self.train_button.pack(side="right", padx=10)

        self.stop_button = ctk.CTkButton(
            self.bottom_button_frame, text="Stop Training", command=self._stop_training
        )
        self.stop_button.pack(side="right", padx=10)
