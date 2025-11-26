import customtkinter as ctk
import threading

from config import LAYER_PARAMS, App
from core.model import BaseModel, get_layer_output_shapes

import torch
import torch.nn as nn


class TrainFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

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
                while shape_idx < len(shapes) and "auto" in shapes[shape_idx][0].lower():
                    shape_idx += 1

                if shape_idx < len(shapes):
                    layer_desc, shape = shapes[shape_idx]
                    if len(shape) == 2:
                        shape_text = f" → {shape[1]} features"
                    elif len(shape) == 4:
                        shape_text = f" → {shape[1]}ch × {shape[2]}×{shape[3]}"
                    shape_idx += 1

                    # Skip activation shapes
                    while shape_idx < len(shapes) and shapes[shape_idx][0].startswith("  └─"):
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
                font=ctk.CTkFont(size=16, weight="bold")
            )
            header.grid(row=0, column=0, columnspan=2, padx=5, pady=(10, 20), sticky="w")

            if not param_template:
                # No parameters for this layer type (e.g., Flatten)
                msg = ctk.CTkLabel(self.param_panel, text="No configurable parameters")
                msg.grid(row=1, column=0, columnspan=2, padx=5, pady=10)
                return

            row_number = 1

            for param_name, param_info in param_template.items():
                # Get current value from the layer (use default if not set)
                param_value = current_layer["params"].get(param_name, param_info["default"])

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
                        lambda event, pn=param_name: self._update_param(pn, event.widget.get()),
                    )
                    # Also update on focus out
                    param_entry.bind(
                        "<FocusOut>",
                        lambda event, pn=param_name: self._update_param(pn, event.widget.get()),
                    )

                elif param_info["type"] == "dropdown":
                    param_combo = ctk.CTkComboBox(
                        self.param_panel, values=param_info["options"]
                    )
                    param_combo.set(param_value if param_value else param_info["options"][0])
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
            self.model_buttons_frame, text="New Model"
        )
        self.new_model_button.grid(row=0, column=0, padx=5, pady=(5, 0), sticky="w")

        self.save_button = ctk.CTkButton(self.model_buttons_frame, text="Save Model")
        self.save_button.grid(row=0, column=1, padx=5, pady=(5, 0), sticky="w")

        self.reset_button = ctk.CTkButton(self.model_buttons_frame, text="Reset Model")
        self.reset_button.grid(row=0, column=2, padx=5, pady=(5, 0), sticky="w")

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
        """Setup hyperparameter input boxes"""
        self.hyperparameters_frame = ctk.CTkFrame(self)
        self.hyperparameters_frame.grid(
            row=self.current_row, column=0, columnspan=2, padx=15, pady=10, sticky="ew"
        )

        self.hyperparameters_frame.grid_columnconfigure(1, weight=1)

        self.learning_rate_label = ctk.CTkLabel(
            self.hyperparameters_frame, text="Learning Rate"
        )
        self.learning_rate_label.grid(row=0, column=0, padx=0, pady=10, sticky="w")

        self.learning_rate_entry = ctk.CTkEntry(
            self.hyperparameters_frame, placeholder_text="0.001"
        )
        self.learning_rate_entry.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        self.current_row += 1

        self.epochs_label = ctk.CTkLabel(self.hyperparameters_frame, text="Epochs")
        self.epochs_label.grid(row=1, column=0, padx=0, pady=10, sticky="w")

        self.epochs_entry = ctk.CTkEntry(
            self.hyperparameters_frame, placeholder_text="10"
        )
        self.epochs_entry.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

        self.current_row += 1

        self.batch_size_label = ctk.CTkLabel(
            self.hyperparameters_frame, text="Batch Size"
        )
        self.batch_size_label.grid(row=2, column=0, padx=0, pady=10, sticky="w")

        self.batch_size_entry = ctk.CTkEntry(
            self.hyperparameters_frame, placeholder_text="32"
        )
        self.batch_size_entry.grid(row=2, column=1, padx=10, pady=10, sticky="ew")

        self.current_row += 1

        self.optimizer_label = ctk.CTkLabel(
            self.hyperparameters_frame, text="Optimizer"
        )
        self.optimizer_label.grid(row=3, column=0, padx=0, pady=10, sticky="w")

        self.optimizer_entry = ctk.CTkComboBox(
            self.hyperparameters_frame, values=App.OPTIMIZER_LIST
        )
        self.optimizer_entry.grid(row=3, column=1, padx=10, pady=10, sticky="ew")

        self.current_row += 1

    def _build_model(self):
        """Build the ML model based on the user inputs"""

        # Get hyperparameters from UI entries
        hyperparams = {
            "learning_rate": float(self.learning_rate_entry.get() or 0.001),
            "epochs": int(self.epochs_entry.get() or 10),
            "batch_size": int(self.batch_size_entry.get() or 32),
            "optimizer": self.optimizer_entry.get() or "Adam"
        }

        # Create model with auto-calculated sizes
        model = BaseModel(self.layers)

        # Create optimizer
        optimizer = model.get_optimizer(hyperparams["optimizer"], hyperparams["learning_rate"])

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
        if not self.layers:
            self._update_display(self.loss_display, "Error: Add layers first!")
            return

        # Clear display areas before starting
        self.epochs_display.delete("1.0", "end")
        self.loss_display.delete("1.0", "end")
        self.accuracy_display.delete("1.0", "end")

        training_thread = threading.Thread(target=self._training_loop)
        training_thread.daemon = True
        training_thread.start()

    def _training_loop(self):
        """Training loop for ML model"""
        try:
            # Build model
            model, optimizer, hyperparams = self._build_model()

            # Print model summary
            print("\n" + "="*60)
            model.summary()
            print("="*60 + "\n")

            # Set device (use GPU if available)
            device = torch.device("cpu")
            model.to(device)

            # Load data
            train_loader = self._get_data_loader(hyperparams["batch_size"])

            # Loss function
            criterion = nn.CrossEntropyLoss()

            self.training_status = True

            # Training loop
            for epoch in range(hyperparams["epochs"]):
                if not self.training_status:
                    break

                model.train()
                total_loss = 0
                correct = 0
                total = 0

                for batch_x, batch_y in train_loader:
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

                    # Track training metrics
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

                # Update metrics after each epoch
                self.current_epoch = epoch + 1
                self.current_loss = total_loss / len(train_loader)
                self.current_accuracy = 100 * correct / total

                # Update display using thread-safe method
                self._update_display(self.epochs_display, f"Epoch: {self.current_epoch}/{hyperparams['epochs']}")
                self._update_display(self.loss_display, f"Loss: {self.current_loss:.4f}")
                self._update_display(self.accuracy_display, f"Accuracy: {self.current_accuracy:.2f}%")

            self.training_status = False

        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            self._update_display(self.loss_display, f"Error: {str(e)}")
            self.training_status = False

    def _get_data_loader(self, batch_size):
        """Get training data"""
        from torchvision import transforms
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = ImageFolder(
            root='mnist_dataset/training',
            transform=transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        return train_loader

    def _get_test_loader(self, batch_size):
        """Get the test data"""
        from torchvision import transforms
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader

        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        test_dataset = ImageFolder(
            root='mnist_dataset/testing',
            transform=transform
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        return test_loader

    def _stop_training(self):
        self.training_status = False

    def setup_plot(self):
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.grid(
            row=0, column=3, padx=10, pady=10, sticky="nsew", rowspan=self.current_row
        )

        self.plot_label = ctk.CTkLabel(self.plot_frame, text="Model Performance")
        self.plot_label.grid(row=0, column=0, padx=10, pady=10)

        self.epochs_display = ctk.CTkTextbox(self.plot_frame, height=50)
        self.epochs_display.grid(row=1, column=0, padx=10, pady=10)

        self.loss_display = ctk.CTkTextbox(self.plot_frame, height=50)
        self.loss_display.grid(row=2, column=0, padx=10, pady=10)

        self.accuracy_display = ctk.CTkTextbox(self.plot_frame, height=50)
        self.accuracy_display.grid(row=3, column=0, padx=10, pady=10)

    def setup_bottom_buttons(self):
        self.bottom_button_frame = ctk.CTkFrame(self)
        self.bottom_button_frame.grid(
            row=self.current_row, column=0, columnspan=5, padx=10, pady=10, sticky="we"
        )

        self.bottom_button_frame.grid_columnconfigure(0, weight=1)

        self.current_row += 1

        self.train_button = ctk.CTkButton(self.bottom_button_frame, text="Train Model", command=self._start_training)
        self.train_button.pack(side="right", padx=10)

        self.stop_button = ctk.CTkButton(self.bottom_button_frame, text="Stop Training", command=self._stop_training)
        self.stop_button.pack(side="right", padx=10)
