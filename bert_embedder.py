"""
Production-ready BERT embedding module.
Uses transformers library for text encoding without dummy models.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from transformers import BertTokenizer, BertModel

logger = logging.getLogger(__name__)


class BertEmbedder:
    """Production BERT embedder with proper model loading."""
    
    def __init__(self, model_dir: str):
        """
        Initialize BERT embedder from local directory.
        
        Args:
            model_dir: Directory containing BERT model files
            
        Raises:
            FileNotFoundError: If model files not found
            RuntimeError: If model initialization fails
        """
        self.model_dir = Path(model_dir)
        
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Validate required files
        required_files = ['config.json', 'vocab.txt']
        missing_files = [f for f in required_files if not (self.model_dir / f).exists()]
        
        if missing_files:
            raise FileNotFoundError(
                f"Missing required files in {model_dir}: {missing_files}"
            )
        
        # Check for model weights
        has_weights = (
            (self.model_dir / 'pytorch_model.bin').exists() or
            (self.model_dir / 'model.safetensors').exists()
        )
        
        if not has_weights:
            raise FileNotFoundError(
                f"No model weights found in {model_dir}. "
                "Expected pytorch_model.bin or model.safetensors"
            )
        
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")
            
            # Load tokenizer
            logger.info(f"Loading BERT tokenizer from {model_dir}")
            self.tokenizer = BertTokenizer.from_pretrained(
                str(self.model_dir),
                local_files_only=True
            )
            
            # Load model
            logger.info(f"Loading BERT model from {model_dir}")
            self.model = BertModel.from_pretrained(
                str(self.model_dir),
                local_files_only=True
            ).to(self.device)
            
            self.model.eval()
            self.hidden_size = self.model.config.hidden_size
            
            logger.info(f"BERT embedder initialized successfully (hidden_size: {self.hidden_size})")
            
        except Exception as e:
            logger.error(f"Failed to initialize BERT embedder: {str(e)}")
            raise RuntimeError(f"BERT initialization failed: {str(e)}") from e
    
    def encode(self, text: str, max_length: int = 512) -> np.ndarray:
        """
        Encode text into BERT embedding vector.
        
        Args:
            text: Input text to encode
            max_length: Maximum sequence length
            
        Returns:
            Mean-pooled embedding vector
            
        Raises:
            ValueError: If text is empty or invalid
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Mean pooling
            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            attention_mask_expanded = attention_mask.unsqueeze(-1).float()
            masked_embeddings = last_hidden_state * attention_mask_expanded
            summed_embeddings = masked_embeddings.sum(dim=1)
            token_counts = attention_mask_expanded.sum(dim=1).clamp(min=1e-9)
            mean_pooled = summed_embeddings / token_counts
            
            embedding_vector = mean_pooled.squeeze(0).cpu().numpy()
            
            return embedding_vector
            
        except Exception as e:
            logger.error(f"Failed to encode text: {str(e)}")
            raise
    
    def encode_batch(self, texts: list, max_length: int = 512) -> np.ndarray:
        """
        Encode multiple texts in batch.
        
        Args:
            texts: List of input texts
            max_length: Maximum sequence length
            
        Returns:
            Array of embedding vectors (num_texts, hidden_size)
        """
        if not texts or not isinstance(texts, list):
            raise ValueError("texts must be a non-empty list")
        
        try:
            # Tokenize batch
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Mean pooling
            last_hidden_state = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            attention_mask_expanded = attention_mask.unsqueeze(-1).float()
            masked_embeddings = last_hidden_state * attention_mask_expanded
            summed_embeddings = masked_embeddings.sum(dim=1)
            token_counts = attention_mask_expanded.sum(dim=1).clamp(min=1e-9)
            mean_pooled = summed_embeddings / token_counts
            
            embedding_vectors = mean_pooled.cpu().numpy()
            
            return embedding_vectors
            
        except Exception as e:
            logger.error(f"Failed to encode batch: {str(e)}")
            raise
        