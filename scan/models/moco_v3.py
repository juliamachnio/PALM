import torch
import torch.nn as nn
import torchvision.models as models

class MoCoV3(nn.Module):
    def __init__(self, backbone='resnet50', dim=256, pred_dim=512):
        """
        MoCo v3 Model.
        Args:
            backbone (str): Backbone architecture ('resnet50' or 'vit' if using ViTs).
            dim (int): Feature dimension.
            pred_dim (int): Predictor MLP dimension.
        """
        super(MoCoV3, self).__init__()

        # Define backbone (default: ResNet50)
        self.backbone = models.resnet50(pretrained=False)
        self.backbone.fc = nn.Identity()  # Remove classifier

        # Define projector head (MLP)
        self.projector = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, dim)
        )

        # Define predictor head (MLP) - only used in training
        self.predictor = nn.Sequential(
            nn.Linear(dim, pred_dim),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(),
            nn.Linear(pred_dim, dim)
        )

    def forward(self, x, return_projector=False):
        """
        Forward pass.
        Args:
            x (Tensor): Input images.
            return_projector (bool): If True, return projected features.
        Returns:
            If training -> returns both projector and predictor outputs.
            If inference -> returns only backbone features or projector features.
        """
        features = self.backbone(x)
        projected = self.projector(features)

        if return_projector:
            return projected

        predicted = self.predictor(projected)
        return projected, predicted