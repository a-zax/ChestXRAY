import torch
import torch.nn as nn

class RCTA(nn.Module):
    """
    Module 3: Triangular Cognitive Attention (RCTA)
    Implements verification loop: Image -> Context -> Hypothesis -> Image
    """
    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super().__init__()
        
        # Three attention layers
        # batch_first=True makes IO [Batch, Seq, Feature]
        self.img_to_context = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.context_to_hypothesis = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.hypothesis_to_img = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Layer norms for residual connections
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Final Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        
    def forward(self, img_feat, clinical_feat, disease_feat):
        """
        Args:
            img_feat: [B, 1, 512] or [B, N, 512] (Visual features)
            clinical_feat: [B, 1, 512] (Clinical context embeddings)
            disease_feat: [B, 1, 512] (Disease prediction embeddings)
            
        Returns:
             verified_feat: [B, 1, 512] (Cognitively aligned features for decoder)
        """
        # Ensure inputs are 3D [B, Seq, D]
        if clinical_feat.dim() == 2:
            clinical_feat = clinical_feat.unsqueeze(1)
        if disease_feat.dim() == 2:
            disease_feat = disease_feat.unsqueeze(1)
        if img_feat.dim() == 2:
            img_feat = img_feat.unsqueeze(1)
            
        # --- Step 1: Image queries Clinical Context ---
        # "What does the history say given these visual findings?"
        # Query: Image, Key/Value: Clinical Text
        context_aware_img, _ = self.img_to_context(
            query=img_feat,
            key=clinical_feat,
            value=clinical_feat
        )
        # Residual
        context_aware_img = self.norm1(context_aware_img + img_feat)
        
        # --- Step 2: Context queries Disease Predictions ---
        # "Given the context-aware image, what diseases are relevant?"
        # Query: Context-Aware Image, Key/Value: Disease Predictions
        hypothesis_aware, _ = self.context_to_hypothesis(
            query=context_aware_img,
            key=disease_feat,
            value=disease_feat
        )
        hypothesis_aware = self.norm2(hypothesis_aware + context_aware_img)
        
        # --- Step 3: Hypothesis queries Image (Verification) ---
        # "Verify these hypotheses against the original image"
        # Query: Hypothesis, Key/Value: Original Image Features
        verified, _ = self.hypothesis_to_img(
            query=hypothesis_aware,
            key=img_feat,
            value=img_feat
        )
        verified = self.norm3(verified + hypothesis_aware)
        
        # Feed-forward network
        output = self.ffn(verified)
        output = self.norm_ffn(output + verified)
        
        return output
