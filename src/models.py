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


class COCO2_Combined_Shared_Partial_Heads(nn.Module):
    """
    Hybrid Shared + Partial Heads model for multi-molecule/isotope regression.
    """
    def __init__(self, input_dim, n_molecules, n_isos, shared_dim=128, iso_head_dim=64):
        super().__init__()
        self.mol_embed = nn.Embedding(n_molecules, 8)
        self.iso_embed = nn.Embedding(n_isos, 8)

        # Input dim + mol_embedding (8) + iso_embedding (8)
        total_dim = input_dim + 8 + 8
        
        self.trunk = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
        )

        # Shared conditional trunk head
        self.shared_head = nn.Sequential(
            nn.Linear(shared_dim + 8, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # Lightweight per-isotope adapters (fine-tuning layer)
        self.iso_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, iso_head_dim),
                nn.GELU(),
                nn.Linear(iso_head_dim, 1)
            ) for _ in range(n_isos)
        ])

        # Gating mechanism: learn per-isotope mixing coefficient (0–1)
        self.gate = nn.Sequential(
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x, mol_idx, iso_idx):
        # Ensure indices are passed as long tensors
        mol_idx = mol_idx.long()
        iso_idx = iso_idx.long()

        mol_emb = self.mol_embed(mol_idx)
        iso_emb = self.iso_embed(iso_idx)

        # Shared trunk representation
        # Concatenate input features with embeddings
        z = torch.cat([x, mol_emb, iso_emb], dim=1)
        shared = self.trunk(z)

        # Shared conditional prediction
        shared_pred = self.shared_head(torch.cat([shared, iso_emb], dim=1))

        # Per-isotope specialized adapter prediction
        out = torch.zeros_like(shared_pred)
        
        # We iterate through heads. 
        # Note: This loop assumes iso_idx corresponds to the index in iso_heads.
        # The data_loader must ensure iso_idx is mapped to 0..N-1.
        for i, head in enumerate(self.iso_heads):
            mask = (iso_idx == i)
            if mask.any():
                # We squeeze the mask to match shared dimensions if necessary, 
                # but here mask is usually 1D (Batch).
                out[mask] = head(shared[mask])

        # Gate determines how much to trust isotope-specific vs shared
        gate_val = self.gate(iso_emb)
        final_pred = gate_val * out + (1 - gate_val) * shared_pred

        # Return shape (N, 1) to match framework convention
        return final_pred 

    def init_output_bias(self, bias_value: float = 0.0):
        # Initialize bias in the trunk's last linear layer is less relevant 
        # due to the heads, but we can try to init the heads or shared_head.
        # The original code inited the trunk, which might be a bug or intended 
        # for the latent space. 
        # Let's initialize the final linear layers of the heads and shared_head.
        
        with torch.no_grad():
            # Shared head
            self.shared_head[-1].bias.fill_(float(bias_value))
            # Iso heads
            for head in self.iso_heads:
                head[-1].bias.fill_(float(bias_value))


# === Model Factory ===
# A dictionary mapping model names (from config) to their classes
MODELS = {
    "CO2_Basic": CO2_Single_Trunk,
    "CO_CO2_Combined": COCO2_Combined_Shared_Partial_Heads,
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
