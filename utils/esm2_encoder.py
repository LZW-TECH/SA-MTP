import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer
import numpy as np


class ESM2Encoder:
    """
    ESM-2 (650M) encoder for protein sequences
    Returns per-residue representations of dimension 1280
    """
    
    def __init__(self, model_name="facebook/esm2_t33_650M_UR50D", device='cuda'):
        """
        Args:
            model_name: ESM-2 model name from HuggingFace (default: 650M model with H=1280)
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name).to(device)
        self.model.eval()  # Set to evaluation mode
        
        print(f"Loaded ESM-2 model: {model_name}")
        print(f"Hidden dimension: {self.model.config.hidden_size}")
    
    @torch.no_grad()
    def encode_sequences(self, sequences):
        """
        Encode a list of protein sequences using ESM-2
        
        Args:
            sequences: List of protein sequences (strings)
        
        Returns:
            embeddings: List of numpy arrays, each of shape (L, 1280) where L is sequence length
        """
        embeddings = []
        
        for seq in sequences:
            # Tokenize
            inputs = self.tokenizer(seq, return_tensors="pt", padding=False, truncation=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            outputs = self.model(**inputs)
            # outputs.last_hidden_state: (1, L+2, 1280) - includes <CLS> and <EOS> tokens
            
            # Remove <CLS> (first) and <EOS> (last) tokens to get only residue embeddings
            residue_embeddings = outputs.last_hidden_state[0, 1:-1, :]  # (L, 1280)
            
            embeddings.append(residue_embeddings.cpu().numpy())
        
        return embeddings
    
    def batch_encode_sequences(self, sequences, batch_size=8):
        """
        Encode sequences in batches for efficiency
        
        Args:
            sequences: List of protein sequences
            batch_size: Number of sequences to process at once
        
        Returns:
            embeddings: List of numpy arrays, each of shape (L, 1280)
        """
        all_embeddings = []
        
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            batch_embeddings = self.encode_sequences(batch_seqs)
            all_embeddings.extend(batch_embeddings)
            
            if (i + batch_size) % 100 == 0:
                print(f"Processed {min(i+batch_size, len(sequences))}/{len(sequences)} sequences")
        
        return all_embeddings


class ESM2ProjectionLayer(nn.Module):
    """
    Projection layer to map ESM-2 features (1280D) to model dimension (256D)
    Architecture: 1280 → 512 → 256
    """
    
    def __init__(self, esm_dim=1280, hidden_dim=512, output_dim=256, dropout=0.1):
        """
        Args:
            esm_dim: ESM-2 output dimension (1280 for 650M model)
            hidden_dim: Hidden dimension (512)
            output_dim: Final output dimension (256)
            dropout: Dropout rate
        """
        super(ESM2ProjectionLayer, self).__init__()
        
        self.projection = nn.Sequential(
            # First projection: 1280 → 512
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            # Second projection: 512 → 256
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout)
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        for module in self.projection.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Args:
            x: ESM-2 features of shape (batch_size, seq_len, 1280)
        
        Returns:
            Projected features of shape (batch_size, seq_len, 256)
        """
        return self.projection(x)

