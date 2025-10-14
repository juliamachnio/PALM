import torch
import torch.nn as nn
import torchvision.models as models


class MLPHead(nn.Module):
    """MLP Head for BYOL (Projector & Predictor)"""

    def __init__(self, in_dim=2048, out_dim=256, hidden_dim=4096):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class BYOL(nn.Module):
    """Bootstrap Your Own Latent (BYOL) Model"""

    def __init__(self, backbone='resnet50', feature_dim=128, predictor_dim=4096, momentum=0.996):
        super().__init__()

        # Online and Target Encoders (ResNet50)
        self.online_encoder = models.resnet50(pretrained=False)
        self.online_encoder.fc = nn.Identity()  # Remove classifier

        self.target_encoder = models.resnet50(pretrained=False)
        self.target_encoder.fc = nn.Identity()

        # Freeze target encoder weights
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Projection and Prediction Heads
        self.projector = MLPHead(in_dim=2048, out_dim=feature_dim, hidden_dim=predictor_dim)
        self.predictor = MLPHead(in_dim=feature_dim, out_dim=feature_dim, hidden_dim=predictor_dim)

        # Target Momentum Parameter
        self.momentum = momentum

    def forward(self, x1, x2):
        """Computes BYOL loss."""
        # Compute online features
        online_proj1 = self.projector(self.online_encoder(x1))
        online_proj2 = self.projector(self.online_encoder(x2))

        # Compute predictions
        pred1 = self.predictor(online_proj1)
        pred2 = self.predictor(online_proj2)

        # Compute target features (without gradients)
        with torch.no_grad():
            target_proj1 = self.projector(self.target_encoder(x1))
            target_proj2 = self.projector(self.target_encoder(x2))

        return pred1, pred2, target_proj1, target_proj2

    @torch.no_grad()
    def update_target_network(self):
        """Momentum update of the target encoder."""
        for online_params, target_params in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = self.momentum * target_params.data + (1 - self.momentum) * online_params.data

    @torch.no_grad()
    def extract_features(self, images, use_projector=False):
        """Extracts features from BYOL."""
        features = self.online_encoder(images)  # Backbone features (2048D)

        if use_projector:
            features = self.projector(features)  # Projector features (256D)

        return features
