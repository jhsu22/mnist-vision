import os
import threading
import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
import PIL.ImageOps
import torch
import torch.nn as nn
from CTkMessagebox import CTkMessagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from config import App, Paths
from core.model import BaseModel


class DemoFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)

        # Configure grid layout
        self.grid_columnconfigure(0, weight=0)  # controls
        self.grid_columnconfigure(1, weight=1)  # output
        self.grid_rowconfigure(0, weight=1)

        # Track current model
        self.current_model = None
        self.current_model_name = None

        # Analysis status
        self.analysis_status = False

        # Initialize analysis variables
        self.results = []
        self.navigation_index = 0

        self.setup_controls()
        self.setup_output()
        self.setup_canvases()
        self.setup_navigation_buttons()

        self._refresh_model_list()

    def setup_controls(self):
        """Left side - control panel"""
        self.controls_frame = ctk.CTkFrame(self, width=300)
        self.controls_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.controls_frame.grid_propagate(False)

        # Title
        title = ctk.CTkLabel(
            self.controls_frame,
            text="Model Demo",
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

        # Load demo data
        load_data_label = ctk.CTkLabel(self.controls_frame, text="Load Demo Data:")
        load_data_label.pack(padx=10, pady=5, anchor="w")

        self.radio_var = tk.IntVar(value=1)
        self.load_option_file = ctk.CTkRadioButton(
            self.controls_frame,
            text="Load Data From File",
            value=1,
            variable=self.radio_var,
            command=self._on_mode_change,
        )
        self.load_option_file.pack(padx=10, pady=5, anchor="w")

        self.directory_select_frame = ctk.CTkFrame(self.controls_frame)
        self.directory_select_frame.pack(padx=10, pady=5, anchor="w")

        self.directory_text = ctk.CTkTextbox(self.directory_select_frame, height=30)
        self.directory_text.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.directory_browse_button = ctk.CTkButton(
            self.directory_select_frame,
            text="Browse",
            command=self._browse_directory,
            width=75,
        )
        self.directory_browse_button.pack(side="right")

        load_option_draw = ctk.CTkRadioButton(
            self.controls_frame,
            text="Draw Data",
            value=2,
            variable=self.radio_var,
            command=self._on_mode_change,
        )
        load_option_draw.pack(padx=10, pady=5, anchor="w")

        # Separator
        separator = ctk.CTkFrame(self.controls_frame, height=2, fg_color="gray30")
        separator.pack(fill="x", padx=20, pady=20)

        # Test Controls
        test_label = ctk.CTkLabel(self.controls_frame, text="Run Test:")
        test_label.pack(padx=10, pady=(10, 5), anchor="w")

        self.run_test_button = ctk.CTkButton(
            self.controls_frame,
            text="Run Analysis",
            command=self._start_analysis,
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
        self.output_frame.grid_rowconfigure(2, weight=0)  # Labels
        self.output_frame.grid_rowconfigure(3, weight=1)  # Plots
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

    def setup_canvases(self):
        self.image_title = ctk.CTkLabel(self.output_frame, text="Image/Drawing")
        self.image_title.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        self.image_canvas = tk.Canvas(
            self.output_frame,
            width=364,
            height=364,
            bg="white",
            highlightthickness=1,
            highlightbackground="gray30",
        )
        self.image_canvas.grid(row=3, column=0, padx=10, pady=10, sticky="nsew")

        plt.style.use("dark_background")

        self.confidence_title = ctk.CTkTextbox(self.output_frame, height=30)
        self.confidence_title.grid(row=2, column=1, padx=10, pady=10, sticky="ew")
        self.confidence_title.insert(0.0, "Model guessed: | xx.xx% Confidence")
        self.confidence_title.configure(state="disabled")

        self.confidence_fig = Figure(figsize=(6, 6), dpi=100, facecolor="#1a1a1a")
        self.ax_confidence = self.confidence_fig.add_subplot(111)

        self.ax_confidence.set_facecolor("#1a1a1a")
        self.ax_confidence.set_title(
            "Model Confidence",
            color="white",
            fontfamily="monospace",
            fontweight="bold",
            fontsize=10,
            pad=10,
        )
        self.ax_confidence.set_xlabel(
            "Digit", color="white", fontfamily="monospace", fontsize=10
        )
        self.ax_confidence.set_ylabel(
            "Confidence (%)", color="white", fontfamily="monospace", fontsize=10
        )
        self.ax_confidence.set_ylim(0, 100)
        self.ax_confidence.tick_params(colors="white", labelsize=9)
        self.ax_confidence.grid(True, alpha=0.2, linestyle="-", color="gray")
        self.ax_confidence.spines["top"].set_visible(False)
        self.ax_confidence.spines["right"].set_visible(False)
        self.ax_confidence.spines["left"].set_color("white")
        self.ax_confidence.spines["bottom"].set_color("white")

        self.ax_confidence.text(
            0.5,
            0.5,
            "Run test to generate chart",
            ha="center",
            va="center",
            transform=self.ax_confidence.transAxes,
            color="gray",
            fontfamily="monospace",
            fontsize=12,
        )

        self.confidence_canvas = FigureCanvasTkAgg(
            self.confidence_fig, master=self.output_frame
        )
        self.confidence_canvas.get_tk_widget().grid(
            row=3, column=1, padx=10, pady=10, sticky="nsew"
        )

    def setup_navigation_buttons(self):
        self.navigation_frame = ctk.CTkFrame(self.output_frame)
        self.navigation_frame.grid(
            row=4, column=0, columnspan=2, padx=0, pady=(0, 10), sticky="nsew"
        )

        self.previous_button = ctk.CTkButton(self.navigation_frame, text="Previous", command=lambda: self._display_result(index=self.navigation_index-1))
        self.previous_button.grid(row=0, column=0, padx=10, pady=5)

        self.next_button = ctk.CTkButton(self.navigation_frame, text="Next", command=lambda: self._display_result(index=self.navigation_index+1))
        self.next_button.grid(row=0, column=1, padx=0, pady=5)

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

    def _update_results(self, text):
        """Update results textbox from any thread (thread-safe)"""

        def update():
            self.output_box.insert("end", text)
            self.output_box.see("end")

        self.after(0, update)

    def _browse_directory(self):
        directory_select = filedialog.askdirectory(initialdir=Paths.DEMO_DIR)

        self.directory_text.delete(0.0, "end")
        self.directory_text.insert(0.0, directory_select)

        return directory_select

    def _on_mode_change(self):
        if self.radio_var.get() == 1:
            # Show directory selection
            self.directory_select_frame.pack(
                after=self.load_option_file, padx=10, pady=5, anchor="w"
            )
        elif self.radio_var.get() == 2:
            # Hide directory selection
            self.directory_select_frame.pack_forget()

    def _start_analysis(self):
        if self.current_model is None:
            CTkMessagebox(
                master=self,
                title="No Model Loaded",
                message="Please load a model first.",
                icon="warning",
            )
            return

        if self.analysis_status:
            CTkMessagebox(
                master=self,
                title="Analysis in Progress",
                message="Analysis is already running.",
                icon="info",
            )
            return

        if self.radio_var.get() == 1:
            self.directory = self.directory_text.get("1.0", "end").strip()

        # Check if directory exists
        if os.path.isdir(self.directory):

            # Clear previous results
            self.output_box.delete("1.0", "end")

            self.results = []

            analysis_thread = threading.Thread(target=self._analysis_loop)
            analysis_thread.daemon = True
            analysis_thread.start()

        else:
            CTkMessagebox(
                master=self,
                title="Invalid Directory",
                message="The specified directory does not exist.",
                icon="warning",
            )

    def _analysis_loop(self):
        self.analysis_status = True
        self.status_label.configure(
            text="Testing in progress...", text_color="orange"
        )

        self._update_results("Scanning photo directory...\n")

        if self.radio_var.get() == 1:

            # Loop through image directory
            for file in os.listdir(self.directory):
                self._update_results(f"Processing image {file} in photo directory...\n")

                # Get only image files
                if file.endswith(".jpg") or file.endswith(".png"):

                    # Get image path of image file
                    image_path = os.path.join(self.directory, file)

                    # Run image analysis
                    self._analyze_image(image_path)

            self._update_results("Analysis complete.\n")
            self.status_label.configure(
                text="Analysis complete.", text_color="green"
            )

            # Reset state variables
            self.analysis_status = False

            self.navigation_index = 0

            if self.results:
                self.after(0, lambda: self._display_result(0))

    def _analyze_image(self, image_path):
        from torchvision import transforms

        # Prepare image for ml model
        transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ]
        )

        # Load image
        image = Image.open(image_path)

        # Transform and convert image to tensor
        tensor = transform(image)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        # Set device
        device = torch.device("cpu")
        self.current_model.to(device)
        self.current_model.eval()

        # Run inference
        self._update_results("Running inference...\n")
        with torch.no_grad():
            output = self.current_model(tensor)

            probabilities = torch.softmax(output, dim=1)

            prediction = torch.argmax(probabilities, dim=1).item()

            confidences = probabilities[0].cpu().numpy() * 100

            self.results.append({
                'image_path': image_path,
                'image': image,
                'prediction': prediction,
                'confidences': confidences.tolist()
            })

        # Update results
        self._update_results(f"Predicted digit: {prediction}\n")

    def _display_result(self, index):

        # Don't run if index is out of range
        if index < 0 or index >= len(self.results):
            return

        # Get result data
        result = self.results[index]

        # Update navigation index
        self.navigation_index = index

        # Get and display image from results
        img = result['image']

        # Get inverse image (black on white bg)
        img = PIL.ImageOps.invert(img)

        # Resize image to fit canvas
        img = img.resize((364, 364))

        # Convert image to Tkinter PhotoImage
        img = ImageTk.PhotoImage(img)

        # Delete previous image from canvas
        self.image_canvas.delete("all")

        # Display image on canvas
        self.image_canvas.create_image(0, 0, image=img, anchor='nw')

        self.image_ref = img

        # Write to confidence title textbox
        self.confidence_title.configure(state="normal")
        self.confidence_title.delete("0.0", "end")
        self.confidence_title.insert("0.0", f"Model predicted: {result['prediction']} with confidence: {np.max(result['confidences']):.2f}%")
        self.confidence_title.configure(state="disabled")

        # Write to confidence plot
        self.ax_confidence.clear()

        digits = np.arange(10)
        self.ax_confidence.bar(
            digits,
            result['confidences'],
            color='#E74C3C',
            edgecolor="white",
            linewidth=0.5
        )

        # Styling
        self.ax_confidence.set_facecolor("#1a1a1a")
        self.ax_confidence.set_title(
            "Model Confidence",
            color="white",
            fontfamily="monospace",
            fontsize=10,
            fontweight="bold",
            pad=10,
        )
        self.ax_confidence.set_xlabel(
            "Digit", color="white", fontfamily="monospace", fontsize=10
        )
        self.ax_confidence.set_ylabel(
            "Confidence (%)", color="white", fontfamily="monospace", fontsize=10
        )

        self.ax_confidence.set_ylim(0, 100)
        self.ax_confidence.set_xticks(digits)
        self.ax_confidence.set_xticklabels(
            ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            fontfamily="monospace",
            fontsize=9,
        )
        self.ax_confidence.tick_params(colors="white", labelsize=9)

        # only horizontal lines
        self.ax_confidence.grid(axis="y", alpha=0.2, linestyle="-", color="gray")

        # Remove top and right spines
        self.ax_confidence.spines["top"].set_visible(False)
        self.ax_confidence.spines["right"].set_visible(False)
        self.ax_confidence.spines["left"].set_color("white")
        self.ax_confidence.spines["bottom"].set_color("white")

        self.ax_confidence.figure.canvas.draw()
