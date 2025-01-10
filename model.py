import torch 
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128*4*4, latent_dim)
        )
    def forward(self,x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128*4*4),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.decoder(x)

class PoseHead(nn.Module):
    def __init__(self, latent_dim, num_keypoints=17, keypoint_dims=3):  # Changed to include visibility flag
        super().__init__()
        self.pose_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_keypoints * keypoint_dims)  # Output: x, y, visibility for each keypoint
        )
    def forward(self,x):
        return self.pose_head(x)
    
class CaptionHead(nn.Module):
    def __init__(self, latent_dim, vocab_size=5000):
        super().__init__()
        self.caption_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, vocab_size)
        )
    def forward(self,x):
        return self.caption_head(x)
