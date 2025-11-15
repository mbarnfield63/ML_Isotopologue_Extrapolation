import torch
import torch.nn as nn


class CO2_Single_Trunk(nn.Module):
    """
    This model consists of a single trunk of fully connected layers with GELU activations
    and Dropout for regularization.
    """

    def __init__(self, input_dim: int, dropout: float = 0.3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.shared(x)

    def init_output_bias(self, bias_value: float = 0.0):
        """
        Set the bias of the final Linear layer to `bias_value`.
        This helps anchor initial predictions to a sensible value (e.g. train mean).
        """
        # last layer is the last module in the sequential (Linear(32,1))
        last_linear = None
        # traverse modules from end to find last nn.Linear
        for module in reversed(list(self.shared)):
            if isinstance(module, nn.Linear):
                last_linear = module
                break
        if last_linear is None:
            raise RuntimeError("No Linear layer found to init bias.")
        with torch.no_grad():
            last_linear.bias.fill_(float(bias_value))


# === Model Factory ===

# A dictionary mapping model names (from config) to their classes
MODELS = {
    "CO2_Basic": CO2_Single_Trunk,
    # Add more models here, e.g.:
    # "SimpleLinearModel": SimpleLinearModel,
}


def get_model(config: dict, input_dim: int):
    """
    Factory function to instantiate a model based on the config.

    Args:
        config (dict): The loaded experiment config.
        input_dim (int): The number of input features, determined by
                         the data_loader.

    Returns:
        torch.nn.Module: An instantiated model.
    """
    model_name = config['model']['name']
    model_params = config['model'].get('params', {})

    if model_name not in MODELS:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Available models are: {list(MODELS.keys())}"
        )

    # Get the model class
    model_class = MODELS[model_name]

    # Instantiate the model with the dynamic input_dim and params from config
    try:
        model = model_class(input_dim=input_dim, **model_params)
        print(f"Model '{model_name}' instantiated successfully.")
    except TypeError as e:
        print(f"Error instantiating model '{model_name}'.")
        print(f"Check `model.params` in your config against the model's __init__ method.")
        print(f"Error details: {e}")
        raise

    return model
