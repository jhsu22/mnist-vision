import os
import threading

import customtkinter as ctk
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from CTkMessagebox import CTkMessagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from config import App, Paths
from core.model import BaseModel


class TestFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

        # Configure grid layout
        self.grid_columnconfigure(0, weight=0)  # controls
        self.grid_columnconfigure(1, weight=1)  # output
        self.grid_rowconfigure(0, weight=1)

        # Track current model
        self.current_model = None
        self.current_model_name = None

        # Testing status
        self.testing_status = False

        # Setup UI
        self.setup_controls()
        self.setup_output()

        self._refresh_model_list()

        self.setup_plots()

    def setup_controls(self):
        """Left side - control panel"""
        self.controls_frame = ctk.CTkFrame(self, width=300)
        self.controls_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.controls_frame.grid_propagate(False)  # Don't shrink

        # Title
        title = ctk.CTkLabel(
            self.controls_frame,
            text="Model Testing",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        title.pack(padx=10, pady=(10, 20))

        # Model Selection
        model_label = ctk.CTkLabel(self.controls_frame, text="Select Model:")
        model_label.pack(padx=10, pady=(10, 5), anchor="w")

        self.model_selection = ctk.CTkComboBox(
            self.controls_frame, values=App.MODEL_LIST, width=260
        )
        self.model_selection.pack(padx=10, pady=5)

        self.load_model_button = ctk.CTkButton(
            self.controls_frame, text="Load Model", command=self._load_model, width=260
        )
        self.load_model_button.pack(padx=10, pady=5)

        self.refresh_models_button = ctk.CTkButton(
            self.controls_frame,
            text="Refresh Model List",
            command=self._refresh_model_list,
            width=260,
        )
        self.refresh_models_button.pack(padx=10, pady=5)

        # Separator
        separator = ctk.CTkFrame(self.controls_frame, height=2, fg_color="gray30")
        separator.pack(fill="x", padx=20, pady=20)

        # Test Controls
        test_label = ctk.CTkLabel(self.controls_frame, text="Run Test:")
        test_label.pack(padx=10, pady=(10, 5), anchor="w")

        self.run_test_button = ctk.CTkButton(
            self.controls_frame,
            text="Run Test on MNIST Dataset",
            command=self._start_testing,
            width=260,
            height=40,
        )
        self.run_test_button.pack(padx=10, pady=5)

        # Status label
        self.status_label = ctk.CTkLabel(
            self.controls_frame, text="No model loaded", text_color="gray"
        )
        self.status_label.pack(padx=10, pady=(20, 10))

    def setup_output(self):
        """Right side - output panel"""
        self.output_frame = ctk.CTkFrame(self)
        self.output_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.output_frame.grid_rowconfigure(0, weight=0)  # Title
        self.output_frame.grid_rowconfigure(1, weight=0)  # Text results
        self.output_frame.grid_rowconfigure(2, weight=1)  # Plots
        self.output_frame.grid_columnconfigure(0, weight=1)
        self.output_frame.grid_columnconfigure(1, weight=1)

        # Title
        output_title = ctk.CTkLabel(
            self.output_frame,
            text="Test Results",
            font=ctk.CTkFont(size=18, weight="bold"),
        )
        output_title.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="w")

        # Output textbox
        self.output_box = ctk.CTkTextbox(
            self.output_frame, font=ctk.CTkFont(family="Courier", size=12), height=200
        )
        self.output_box.grid(
            row=1, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="nsew"
        )

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
            all_models = ["No models found"]

        # Update model selection dropdown
        self.model_selection.configure(values=all_models)

        if self.current_model_name and self.current_model_name in saved_models:
            self.model_selection.set(self.current_model_name)
        elif saved_models:
            self.model_selection.set(saved_models[0])
        else:
            self.model_selection.set("No models found")

    def _load_model(self):
        """Load a model from a .pth file"""
        model_name = self.model_selection.get()

        if not model_name or model_name == "No models found":
            CTkMessagebox(
                master=self,
                title="Load Model",
                message="No model selected.",
                icon="warning",
            )
            return

        file_path = os.path.join(Paths.SAVED_MODELS_DIR, f"{model_name}.pth")

        if not os.path.exists(file_path):
            CTkMessagebox(
                master=self,
                title="Load Model",
                message=f"Model '{model_name}' not found.",
                icon="cancel",
            )
            return

        try:
            # Load the model
            save_data = torch.load(file_path, weights_only=False)

            # Check for trained weights
            if "model_state_dict" not in save_data:
                CTkMessagebox(
                    master=self,
                    title="Load Model",
                    message=f"Model '{model_name}' has no trained weights.\nPlease train the model first.",
                    icon="warning",
                )
                return

            # Restore layer configuration
            layers = save_data.get("layer_configs", [])

            if layers:
                self.current_model = BaseModel(layers)
                self.current_model.load_state_dict(save_data["model_state_dict"])
                self.current_model.eval()
                self.current_model_name = model_name

                self.status_label.configure(
                    text=f"Model '{model_name}' loaded", text_color="green"
                )

                CTkMessagebox(
                    master=self,
                    title="Load Model",
                    message=f"Model '{model_name}' loaded successfully.",
                    icon="check",
                )
            else:
                self.current_model = None
                CTkMessagebox(
                    master=self,
                    title="Load Model",
                    message="Model has no layer configuration.",
                    icon="warning",
                )

        except Exception as e:
            CTkMessagebox(
                master=self,
                title="Load Model",
                message=f"Error loading model:\n{str(e)}",
                icon="cancel",
            )

    def _start_testing(self):
        """Run testing loop in a separate thread"""
        if self.current_model is None:
            CTkMessagebox(
                master=self,
                title="No Model Loaded",
                message="Please load a model first.",
                icon="warning",
            )
            return

        if self.testing_status:
            CTkMessagebox(
                master=self,
                title="Test in Progress",
                message="Testing is already running.",
                icon="info",
            )
            return

        # Clear previous results
        self.output_box.delete("1.0", "end")

        testing_thread = threading.Thread(target=self._testing_loop)
        testing_thread.daemon = True
        testing_thread.start()

    def _testing_loop(self):
        """Testing loop for MNIST dataset"""
        try:
            self.testing_status = True
            self.status_label.configure(
                text="Testing in progress...", text_color="orange"
            )

            self._update_results("Starting test on MNIST dataset...\n")

            # Set device
            device = torch.device("cpu")
            self.current_model.to(device)
            self.current_model.eval()

            # Load test data
            test_loader = self._get_test_loader(batch_size=32)

            self._update_results(
                f"Test dataset size: {len(test_loader.dataset)} images\n"
            )
            self._update_results("Running inference...\n\n")

            # Initialize metrics
            correct = 0
            total = 0
            class_correct = [0] * 10
            class_total = [0] * 10
            self.confusion_matrix = np.zeros((10, 10))

            # Run inference
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(test_loader):
                    images, labels = images.to(device), labels.to(device)

                    # Forward pass
                    outputs = self.current_model(images)
                    _, predicted = torch.max(outputs.data, 1)

                    # Overall accuracy
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    # Per-class accuracy
                    for i in range(labels.size(0)):
                        label = labels[i]
                        pred = predicted[i]

                        # Add to confusion matrix
                        self.confusion_matrix[label][pred] += 1

                        # Update class total
                        class_total[label] += 1

                        # Update class correct
                        if label == pred:
                            class_correct[label] += 1

                    # Progress update every 10 batches
                    if (batch_idx + 1) % 10 == 0:
                        self._update_results(
                            f"Processed {total}/{len(test_loader.dataset)} images...\n"
                        )

            # Calculate final metrics
            overall_accuracy = 100 * correct / total

            # Build results string
            results = "\n" + "=" * 60 + "\n"
            results += "TEST RESULTS\n"
            results += "=" * 60 + "\n\n"
            results += (
                f"Overall Accuracy: {overall_accuracy:.2f}% ({correct}/{total})\n\n"
            )
            results += "Per-Class Accuracy:\n"
            results += "-" * 60 + "\n"

            self.class_accuracies = []

            for i in range(10):
                if class_total[i] > 0:
                    class_acc = 100 * class_correct[i] / class_total[i]
                    self.class_accuracies.append(class_acc)
                    results += f"Digit {i}: {class_acc:>5.2f}% ({class_correct[i]:>4}/{class_total[i]:>4})\n"
                else:
                    results += f"Digit {i}: No samples\n"

            results += "=" * 60 + "\n"

            self._update_results(results)

            print(f"\nTest Complete - Overall Accuracy: {overall_accuracy:.2f}%\n")

            self.status_label.configure(
                text=f"Test complete: {overall_accuracy:.2f}% accuracy",
                text_color="green",
            )

            self._update_plots()

        except Exception as e:
            import traceback

            error_msg = f"\nError during testing:\n{str(e)}\n{traceback.format_exc()}"
            self._update_results(error_msg)
            print(error_msg)
            self.status_label.configure(text="Test failed", text_color="red")

        finally:
            self.testing_status = False

    def _update_results(self, text):
        """Update results textbox from any thread (thread-safe)"""

        def update():
            self.output_box.insert("end", text)
            self.output_box.see("end")

        self.after(0, update)

    def setup_plots(self):
        """Setup confusion matrix and accuracy graph"""

        # Create figures
        plt.style.use("dark_background")
        self.confusion_matrix_fig = Figure(figsize=(6, 6), dpi=100, facecolor="#1a1a1a")
        self.accuracy_fig = Figure(figsize=(6, 6), dpi=100, facecolor="#1a1a1a")

        # Create subplots
        self.ax_confusion_matrix = self.confusion_matrix_fig.add_subplot(111)
        self.ax_accuracy = self.accuracy_fig.add_subplot(111)

        # Initialize with empty plots
        self.ax_confusion_matrix.set_facecolor("#1a1a1a")
        self.ax_confusion_matrix.set_title(
            "Confusion Matrix",
            color="white",
            fontfamily="monospace",
            fontweight="bold",
            fontsize=10,
            pad=10,
        )
        self.ax_confusion_matrix.set_xlabel(
            "Predicted", color="white", fontfamily="monospace", fontsize=10
        )
        self.ax_confusion_matrix.set_ylabel(
            "Actual", color="white", fontfamily="monospace", fontsize=10
        )
        self.ax_confusion_matrix.tick_params(colors="white", labelsize=9)
        self.ax_confusion_matrix.text(
            0.5,
            0.5,
            "Run test to generate matrix",
            ha="center",
            va="center",
            transform=self.ax_confusion_matrix.transAxes,
            color="gray",
            fontfamily="monospace",
            fontsize=12,
        )

        self.ax_accuracy.set_facecolor("#1a1a1a")
        self.ax_accuracy.set_title(
            "Per-Class Accuracy",
            color="white",
            fontfamily="monospace",
            fontweight="bold",
            fontsize=10,
            pad=10,
        )
        self.ax_accuracy.set_xlabel(
            "Digit", color="white", fontfamily="monospace", fontsize=10
        )
        self.ax_accuracy.set_ylabel(
            "Accuracy (%)", color="white", fontfamily="monospace", fontsize=10
        )
        self.ax_accuracy.tick_params(colors="white", labelsize=9)
        self.ax_accuracy.grid(True, alpha=0.2, linestyle="-", color="gray")
        self.ax_accuracy.spines["top"].set_visible(False)
        self.ax_accuracy.spines["right"].set_visible(False)
        self.ax_accuracy.spines["left"].set_color("white")
        self.ax_accuracy.spines["bottom"].set_color("white")
        self.ax_accuracy.set_ylim(0, 100)
        self.ax_accuracy.text(
            0.5,
            0.5,
            "Run test to generate chart",
            ha="center",
            va="center",
            transform=self.ax_accuracy.transAxes,
            color="gray",
            fontfamily="monospace",
            fontsize=12,
        )

        # Create canvases
        self.confusion_canvas = FigureCanvasTkAgg(
            self.confusion_matrix_fig, master=self.output_frame
        )
        self.confusion_canvas.get_tk_widget().grid(
            row=2, column=0, padx=10, pady=10, sticky="nsew"
        )

        self.accuracy_canvas = FigureCanvasTkAgg(
            self.accuracy_fig, master=self.output_frame
        )
        self.accuracy_canvas.get_tk_widget().grid(
            row=2, column=1, padx=10, pady=10, sticky="nsew"
        )

        # Make plot row expandable
        self.output_frame.grid_rowconfigure(2, weight=1)

    def _update_plots(self):
        """Update confusion matrix and accuracy plots with test results"""

        # Clear previous plots
        self.ax_confusion_matrix.clear()
        self.ax_accuracy.clear()

        # --- CONFUSION MATRIX ---
        im = self.ax_confusion_matrix.imshow(
            self.confusion_matrix, cmap="Reds", aspect="auto"
        )

        # Add colorbar
        cbar = self.confusion_matrix_fig.colorbar(im, ax=self.ax_confusion_matrix)
        cbar.ax.tick_params(labelsize=9, colors="white")

        # Set ticks and labels
        self.ax_confusion_matrix.set_xticks(np.arange(10))
        self.ax_confusion_matrix.set_xticklabels(
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            fontfamily="monospace",
            fontsize=9,
        )
        self.ax_confusion_matrix.set_yticks(np.arange(10))
        self.ax_confusion_matrix.set_yticklabels(
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            fontfamily="monospace",
            fontsize=9,
        )

        # Styling
        self.ax_confusion_matrix.set_facecolor("#1a1a1a")
        self.ax_confusion_matrix.set_title(
            "Confusion Matrix",
            color="white",
            fontfamily="monospace",
            fontsize=10,
            fontweight="bold",
            pad=10,
        )
        self.ax_confusion_matrix.set_xlabel(
            "Predicted", color="white", fontfamily="monospace", fontsize=10
        )
        self.ax_confusion_matrix.set_ylabel(
            "Actual", color="white", fontfamily="monospace", fontsize=10
        )
        self.ax_confusion_matrix.tick_params(colors="white")

        self.confusion_matrix_fig.tight_layout()

        # --- ACCURACY BAR CHART ---
        digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        bars = self.ax_accuracy.bar(
            digits,
            self.class_accuracies,
            color="#E74C3C",
            edgecolor="white",
            linewidth=0.5,
        )

        # Styling
        self.ax_accuracy.set_facecolor("#1a1a1a")
        self.ax_accuracy.set_xlabel(
            "Digit", color="white", fontfamily="monospace", fontsize=10
        )
        self.ax_accuracy.set_ylabel(
            "Accuracy (%)", color="white", fontfamily="monospace", fontsize=10
        )
        self.ax_accuracy.set_title(
            "Per-Class Accuracy",
            color="white",
            fontfamily="monospace",
            fontsize=10,
            fontweight="bold",
            pad=10,
        )
        self.ax_accuracy.set_ylim(0, 105)  # Slight padding at top
        self.ax_accuracy.set_xticks(digits)
        self.ax_accuracy.set_xticklabels(
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            fontfamily="monospace",
            fontsize=9,
        )
        self.ax_accuracy.tick_params(colors="white", labelsize=9)

        # only horizontal lines
        self.ax_accuracy.grid(axis="y", alpha=0.2, linestyle="-", color="gray")

        # Remove top and right spines
        self.ax_accuracy.spines["top"].set_visible(False)
        self.ax_accuracy.spines["right"].set_visible(False)
        self.ax_accuracy.spines["left"].set_color("white")
        self.ax_accuracy.spines["bottom"].set_color("white")

        # Add percentage labels on top of bars
        for i, (digit, acc) in enumerate(zip(digits, self.class_accuracies)):
            self.ax_accuracy.text(
                digit,
                acc + 1.5,
                f"{acc:.1f}%",
                ha="center",
                va="bottom",
                color="white",
                fontfamily="monospace",
                fontsize=8,
            )

        self.accuracy_fig.tight_layout()

        # Redraw canvases
        self.confusion_canvas.draw()
        self.accuracy_canvas.draw()
