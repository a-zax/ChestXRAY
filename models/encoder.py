import torch
import torch.nn as nn
import timm

class PROFA(nn.Module):
    """
    Module 1: Hierarchical Visual Alignment (PRO-FA)
    Extracts multi-scale visual features: Pixel-level, Region-level, and Organ-level.
    """
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        # Load pretrained ViT
        # We need the features, not the classifier
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get feature dimension (e.g., 768 for ViT-B)
        self.feat_dim = self.vit.embed_dim if hasattr(self.vit, 'embed_dim') else self.vit.num_features
        
        # Projection layers for different granularities
        self.pixel_proj = nn.Linear(self.feat_dim, 512)
        self.region_proj = nn.Linear(self.feat_dim, 512)
        self.organ_proj = nn.Linear(self.feat_dim, 512)
        
        # Layer norms for stability
        self.layer_norm_organ = nn.LayerNorm(512)
        self.layer_norm_region = nn.LayerNorm(512)
        self.layer_norm_pixel = nn.LayerNorm(512)
        
    def forward(self, images):
        """
        Args:
            images: [Batch, 3, 224, 224]
        Returns:
            dict of features: 'pixel', 'region', 'organ'
        """
        # Extract features from ViT
        # forward_features typically returns [B, N, D] including CLS token
        features = self.vit.forward_features(images)  # [B, 197, 768] for ViT-B/16
        
        # Separate [CLS] token and patch tokens
        # Assuming CLS token is at index 0 (standard for ViT)
        cls_token = features[:, 0]  # [B, 768]
        patch_tokens = features[:, 1:]  # [B, 196, 768]
        
        # --- Organ-level: Use CLS token ---
        organ_feat = self.organ_proj(cls_token)  # [B, 512]
        organ_feat = self.layer_norm_organ(organ_feat)
        
        # --- Region-level: Average pool patches in groups ---
        # ViT-B/16 on 224x224 image results in 14x14 patches (196 total)
        B, N, D = patch_tokens.shape
        H = W = int(N ** 0.5)  # Should be 14
        
        patch_grid = patch_tokens.reshape(B, H, W, D)
        
        # Pool to get region concepts (e.g., 7x7 grid)
        # 14x14 -> 2x2 pooling -> 7x7
        region_tokens = torch.nn.functional.avg_pool2d(
            patch_grid.permute(0, 3, 1, 2),  # [B, D, H, W]
            kernel_size=2,
            stride=2
        )  # [B, D, 7, 7]
        
        region_tokens = region_tokens.flatten(2).transpose(1, 2) # [B, 49, D]
        region_feat = self.region_proj(region_tokens)  # [B, 49, 512]
        region_feat = self.layer_norm_region(region_feat)
        
        # --- Pixel-level (Patch-level): Use all patch tokens ---
        pixel_feat = self.pixel_proj(patch_tokens)  # [B, 196, 512]
        pixel_feat = self.layer_norm_pixel(pixel_feat)
        
        return {
            'pixel': pixel_feat,   # [B, 196, 512]
            'region': region_feat, # [B, 49, 512]
            'organ': organ_feat,   # [B, 512]
            'raw_features': features
        }
