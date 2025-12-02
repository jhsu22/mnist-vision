# config.py


import customtkinter


class Paths:
    """All directory paths used by the application"""

    SAVED_MODELS_DIR = "models"
    DATASET_DIR = "mnist_dataset"
    DEMO_DIR = "assets/demo_example"
    TEST_DIR = "mnist_dataset/testing"
    THEME_PATH = "assets/themes/dark.json"


class App:
    """Application configuration"""

    TITLE = "MNIST Playground"
    GEOMETRY = "1200x800"
    MODEL_LIST = [""]
    OPTIMIZER_LIST = ["Adam", "SGD", "RMSprop"]
    LAYER_LIST = ["Linear", "Conv2D", "MaxPooling2D", "Flatten"]

    FONT_TITLE = None
    FONT_REGULAR = None


class Layers:
    """All layer configurations used by the application"""

    HYPERPARAMETERS = {
        "epochs": 10,
        "learning_rate": 0.001,
        "batch_size": 32,
        "optimizer": "Adam",
        "activation": "relu",
        "loss": "crossentropy",
    }

    LINEAR = {
        "output_size": {"type": "int", "default": 64, "label": "Output Size"},
        "activation": {
            "type": "dropdown",
            "default": "ReLU",
            "options": ["ReLU", "Sigmoid", "Tanh", "None"],
            "label": "Activation",
        },
    }

    CONV2D = {
        "out_channels": {"type": "int", "default": 32, "label": "Output Channels"},
        "kernel_size": {"type": "int", "default": 3, "label": "Kernel Size"},
        "stride": {"type": "int", "default": 1, "label": "Stride"},
        "padding": {"type": "int", "default": 0, "label": "Padding"},
        "activation": {
            "type": "dropdown",
            "default": "ReLU",
            "options": ["ReLU", "Sigmoid", "Tanh", "None"],
            "label": "Activation",
        },
    }

    MAXPOOLING2D = {
        "kernel_size": {"type": "int", "default": 2, "label": "Kernel Size"},
        "stride": {"type": "int", "default": 2, "label": "Stride"},
        "padding": {"type": "int", "default": 0, "label": "Padding"},
    }

    FLATTEN = {}  # Flatten has no parameters


# Map layer parameters
LAYER_PARAMS = {
    "Linear": Layers.LINEAR,
    "Conv2D": Layers.CONV2D,
    "MaxPooling2D": Layers.MAXPOOLING2D,
    "Flatten": Layers.FLATTEN,
}


def initialize_fonts():
    """Initializes the application fonts"""
    App.FONT_TITLE = customtkinter.CTkFont(family="Albert Sans", size=24, weight="bold")
    App.FONT_REGULAR = customtkinter.CTkFont(
        family="Albert Sans", size=14, weight="normal"
    )
