import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import os


class MultiHeadAttentionWithSigmoid(nn.Module):
    """Multi-head attention with sigmoid applied after each head"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into heads: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        
        # CUSTOM MODIFICATION: Apply sigmoid after each head
        attn_output = torch.sigmoid(attn_output)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(attn_output)
        
        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """Feed-forward network with GELU activation"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    """Single transformer encoder layer"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttentionWithSigmoid(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """Single transformer decoder layer"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.self_attn = MultiHeadAttentionWithSigmoid(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttentionWithSigmoid(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Cross-attention
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x


class Transformer(nn.Module):
    """Complete transformer model with sigmoid after attention heads"""
    
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_seq_len=5000,
        dropout=0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Encoder and decoder stacks
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_parameters()
        
    def _init_parameters(self):
        """Initialize parameters with Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def make_src_mask(self, src):
        """Generate mask for source padding"""
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_tgt_mask(self, tgt):
        """Generate mask for target padding and future positions"""
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        tgt_len = tgt.size(1)
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask
    
    def encode(self, src, src_mask=None):
        """Encode source sequence"""
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return x
    
    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        """Decode target sequence"""
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        
        return x
    
    def forward(self, src, tgt):
        """Forward pass through the transformer"""
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        
        enc_output = self.encode(src, src_mask)
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        
        output = self.fc_out(dec_output)
        
        return output


class TextDataset(Dataset):
    """Dataset for text file - creates source-target pairs"""
    
    def __init__(self, text_file, seq_len=50):
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Create character-level vocabulary
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i+1 for i, ch in enumerate(self.chars)}  # 0 reserved for padding
        self.idx_to_char = {i+1: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars) + 1  # +1 for padding
        
        # Encode text
        self.data = [self.char_to_idx[ch] for ch in text]
        self.seq_len = seq_len
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Text length: {len(self.data)} characters")
        print(f"Number of sequences: {len(self.data) - seq_len}")
        
    def __len__(self):
        return max(0, len(self.data) - self.seq_len)
    
    def __getitem__(self, idx):
        # Source: characters at positions idx to idx+seq_len-1
        # Target: characters at positions idx+1 to idx+seq_len (shifted by 1)
        src = torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long)
        tgt = torch.tensor(self.data[idx+1:idx+self.seq_len+1], dtype=torch.long)
        return src, tgt


def get_device():
    """Get the best available device (MPS for Mac, CUDA for NVIDIA, or CPU)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders) for acceleration on Mac")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def train(model, train_loader, criterion, optimizer, device, epoch):
    """Training loop for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for src, tgt in pbar:
        src, tgt = src.to(device), tgt.to(device)
        
        # Forward pass (predict next characters)
        output = model(src, tgt[:, :-1])  # Exclude last target token
        
        # Reshape for loss calculation
        output = output.reshape(-1, output.shape[-1])
        tgt = tgt[:, 1:].reshape(-1)  # Exclude first target token
        
        # Calculate loss
        loss = criterion(output, tgt)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


def generate_text(model, dataset, device, seed_text="The ", length=200):
    """Generate text from the model"""
    model.eval()
    
    # Encode seed text
    encoded = [dataset.char_to_idx.get(ch, 0) for ch in seed_text]
    
    with torch.no_grad():
        for _ in range(length):
            # Prepare input
            src = torch.tensor([encoded], dtype=torch.long).to(device)
            tgt = torch.tensor([encoded], dtype=torch.long).to(device)
            
            # Get prediction
            output = model(src, tgt)
            next_token_logits = output[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            
            # Add to sequence
            encoded.append(next_token)
            
            # Keep only last seq_len tokens
            if len(encoded) > dataset.seq_len:
                encoded = encoded[-dataset.seq_len:]
    
    # Decode to text
    generated = ''.join([dataset.idx_to_char.get(idx, '') for idx in encoded])
    return generated


def main():
    parser = argparse.ArgumentParser(description='Train Transformer with Sigmoid on text file')
    
    # File and data parameters
    parser.add_argument('--text_file', type=str, required=True, help='Path to text file')
    parser.add_argument('--seq_len', type=int, default=50, help='Sequence length')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of encoder/decoder layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='Feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Other parameters
    parser.add_argument('--save_path', type=str, default='model.pt', help='Path to save model')
    parser.add_argument('--seed_text', type=str, default='The square root of 4', help='Seed text for generation')
    parser.add_argument('--gen_length', type=int, default=200, help='Length of generated text')
    
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Get device
    device = get_device()
    
    # Load dataset
    print(f"\nLoading text file: {args.text_file}")
    dataset = TextDataset(args.text_file, seq_len=args.seq_len)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # Create model
    print(f"\nCreating model...")
    model = Transformer(
        src_vocab_size=dataset.vocab_size,
        tgt_vocab_size=dataset.vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_layers,
        num_decoder_layers=args.num_layers,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len * 2,
        dropout=args.dropout
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} parameters")
    print(f"\nTraining configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Sequence length: {args.seq_len}")
    print(f"  Device: {device}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nStarting training...\n")
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        avg_loss = train(model, train_loader, criterion, optimizer, device, epoch)
        print(f"Epoch {epoch}/{args.epochs} - Average Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'vocab_size': dataset.vocab_size,
                'char_to_idx': dataset.char_to_idx,
                'idx_to_char': dataset.idx_to_char,
            }, args.save_path)
            print(f"  → Model saved to {args.save_path}")
        
        # Generate sample text every epoch
        if epoch % 1 == 0:
            print(f"\n  Generated text sample:")
            generated = generate_text(model, dataset, device, args.seed_text, args.gen_length)
            print(f"  '{generated}'\n")
    
    print("\nTraining complete!")
    print(f"\nFinal text generation:")
    generated = generate_text(model, dataset, device, args.seed_text, args.gen_length)
    print(f"\n{generated}\n")


if __name__ == "__main__":
    main()