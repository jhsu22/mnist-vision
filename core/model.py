"""
Base ML model that will receive input configurations from UI
Automatically calculates input/output sizes for Linear layers
"""

import torch
import torch.nn as nn
import torch.optim as optim


class BaseModel(nn.Module):
    def __init__(self, layer_configs, input_shape=(1, 1, 28, 28)):
        """
        Args:
            layer_configs: List of layer configuration dictionaries
            input_shape: Shape of input tensor (batch, channels, height, width)
                        Default is MNIST: (1, 1, 28, 28)
        """
        super().__init__()

        self.layers = nn.ModuleList()

        # Use a dummy tensor to calculate sizes dynamically
        x = torch.zeros(input_shape)

        needs_flatten = False

        for i, config in enumerate(layer_configs):
            layer_type = config["type"]
            layer_params = config.get("params", {})

            # Check if we need to flatten before a Linear layer
            if layer_type == "Linear" and len(x.shape) > 2:
                flatten = nn.Flatten()
                self.layers.append(flatten)

                x = flatten(x)

                needs_flatten = False

            # Build the layer with auto-calculated sizes
            if layer_type == "Linear":
                # Auto-calculate input size from current tensor shape
                in_features = x.shape[1]  # After flatten, shape is (batch, features)
                out_features = layer_params.get("output_size", 64)

                layer = nn.Linear(in_features, out_features)
                self.layers.append(layer)
                x = layer(x)

            elif layer_type == "Conv2D":
                # Auto-calculate input channels from current tensor
                in_channels = x.shape[1]  # Shape is (batch, channels, H, W)
                out_channels = layer_params.get("out_channels", 32)
                kernel_size = layer_params.get("kernel_size", 3)
                stride = layer_params.get("stride", 1)
                padding = layer_params.get("padding", 0)

                layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
                self.layers.append(layer)
                x = layer(x)

            elif layer_type == "MaxPooling2D":
                kernel_size = layer_params.get("kernel_size", 2)
                stride = layer_params.get("stride", 2)
                padding = layer_params.get("padding", 0)

                layer = nn.MaxPool2d(kernel_size, stride, padding)
                self.layers.append(layer)
                x = layer(x)

            elif layer_type == "Flatten":
                layer = nn.Flatten()
                self.layers.append(layer)
                x = layer(x)

            else:
                raise ValueError(f"Unknown layer type: {layer_type}")

            # Add activation function if specified
            if "activation" in layer_params:
                activation = self._get_activation(layer_params["activation"])
                self.layers.append(activation)
                x = activation(x)

        # Store the final output shape for reference
        self.output_shape = x.shape

    def _get_activation(self, name):
        """Get activation function from layer config"""
        activations = {
            "ReLU": nn.ReLU(),
            "Sigmoid": nn.Sigmoid(),
            "Tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh()
        }
        return activations.get(name, nn.ReLU())

    def get_optimizer(self, name, learning_rate):
        """Get optimizer from layer config"""
        optimizers = {
            "Adam": optim.Adam,
            "SGD": optim.SGD,
            "RMSprop": optim.RMSprop
        }
        optimizer_class = optimizers.get(name, optim.Adam)
        return optimizer_class(self.parameters(), lr=learning_rate)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def summary(self):
        """Print a summary of the model architecture"""
        print("Model Summary:")
        print("-" * 60)
        x = torch.zeros(1, 1, 28, 28)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            print(f"{i+1}. {layer.__class__.__name__:20} -> Output shape: {list(x.shape)}")
        print("-" * 60)


def get_layer_output_shapes(layer_configs, input_shape=(1, 1, 28, 28)):
    """
    Calculate output shapes for each layer without building the full model.
    Useful for displaying in the UI.

    Returns:
        List of tuples: [(layer_type, output_shape), ...]
    """
    shapes = []
    x = torch.zeros(input_shape)

    for config in layer_configs:
        layer_type = config["type"]
        layer_params = config.get("params", {})

        # Auto-flatten before Linear if needed
        if layer_type == "Linear" and len(x.shape) > 2:
            x = x.flatten(start_dim=1)
            shapes.append(("Flatten (auto)", list(x.shape)))

        if layer_type == "Linear":
            in_features = x.shape[1]
            out_features = layer_params.get("output_size", 64)
            x = torch.zeros(x.shape[0], out_features)
            shapes.append((f"Linear ({in_features} → {out_features})", list(x.shape)))

        elif layer_type == "Conv2D":
            in_channels = x.shape[1]
            out_channels = layer_params.get("out_channels", 32)
            kernel_size = layer_params.get("kernel_size", 3)
            stride = layer_params.get("stride", 1)
            padding = layer_params.get("padding", 0)

            h_out = (x.shape[2] - kernel_size + 2 * padding) // stride + 1
            w_out = (x.shape[3] - kernel_size + 2 * padding) // stride + 1
            x = torch.zeros(x.shape[0], out_channels, h_out, w_out)
            shapes.append((f"Conv2D ({in_channels} → {out_channels})", list(x.shape)))

        elif layer_type == "MaxPooling2D":
            kernel_size = layer_params.get("kernel_size", 2)
            stride = layer_params.get("stride", 2)
            padding = layer_params.get("padding", 0)

            h_out = (x.shape[2] - kernel_size + 2 * padding) // stride + 1
            w_out = (x.shape[3] - kernel_size + 2 * padding) // stride + 1
            x = torch.zeros(x.shape[0], x.shape[1], h_out, w_out)
            shapes.append((f"MaxPool2D ({kernel_size}x{kernel_size})", list(x.shape)))

        elif layer_type == "Flatten":
            x = x.flatten(start_dim=1)
            shapes.append(("Flatten", list(x.shape)))

        if "activation" in layer_params:
            shapes.append((f"  └─ {layer_params['activation']}", list(x.shape)))

    return shapes
