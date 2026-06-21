import torch
import torch.nn as nn
import math
import os



# 1. OUR TRAINED ARCHITECTURE

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=30000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.size(1) > self.pe.size(1): return x + self.pe[:, :self.pe.size(1)]
        return x + self.pe[:, :x.size(1)]


class AdvancedAudioEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        # Total Stride calculation: 5 * 4 * 4 * 8 = 640
        # 16000Hz / 640 = 25Hz (40ms hop size) -> MATCHES COMPETITION REQ
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=10, stride=5, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv1d(128, 256, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Conv1d(256, 512, kernel_size=16, stride=8, padding=4),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.1),
        )
        self.pos_encoder = PositionalEncoding(512)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(512, 8, 2048, dropout=0.1, batch_first=True),
            num_layers=6
        )
        self.output_proj = nn.Linear(512, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        # x input: [Batch, Time]
        if x.ndim == 2: x = x.unsqueeze(1)  # -> [Batch, 1, Time]

        x = self.conv_layers(x)  # -> [Batch, 512, Time/640]
        x = x.transpose(1, 2)  # -> [Batch, Time/640, 512]
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = self.output_proj(x)  # -> [Batch, Time/640, 128]
        return self.layer_norm(x)


# 2. THE COMPETITION WRAPPER

class MyEncoder(nn.Module):
    def __init__(self, model_path="best_model.pth"):
        super().__init__()
        # REQUIRED ATTRIBUTES
        self.sampling_rate = 16000
        self.output_dim = 128
        self.hop_size_in_ms = 40

        # Initialize architecture
        self.model = AdvancedAudioEncoder(output_dim=self.output_dim)

        # Load Weights (Safely)
        if os.path.exists(model_path):
            try:
                # Load to CPU to avoid CUDA errors on organizer's eval server
                state_dict = torch.load(model_path, map_location='cpu')
                self.model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Warning: Could not load weights: {e}")
        else:
            print("Warning: 'best_model.pth' not found. Using random weights (Debugging only).")

    def forward(self, audio: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        API Requirement:
        Input: audio [B, T], attention_mask [B, T] (optional)
        Output: (embeddings [B, T', D], padding_mask [B, T'])
        """

        # 1. Forward Pass
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        embeddings = self.model(audio)  # Output is [B, T', 128]

        # 2. Handle Mask Alignment
        # our model reduces time dimension by 640x.
        # We must reduce the mask by the same amount so they match.
        out_mask = None
        if attention_mask is not None:
            # attention_mask is [B, T]
            # Reshape for pooling: [B, 1, T]
            mask_float = attention_mask.float().unsqueeze(1)

            # Interpolate to match embedding time dimension
            target_time = embeddings.shape[1]
            out_mask = torch.nn.functional.interpolate(
                mask_float,
                size=target_time,
                mode='nearest'
            ).squeeze(1)  # Back to [B, T']

            # Convert back to boolean (1=Keep, 0=Discard)
            out_mask = out_mask > 0.5

        # Return TUPLE as required by checker
        return embeddings, out_mask


if __name__ == "__main__":
    from audio_encoder_checker import check_audio_encoder

    encoder = MyEncoder("best_model.pth")
    check_audio_encoder(encoder)
    print("Check Passed locally.")
