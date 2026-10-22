import torch
import torch.nn as nn
from transformers import (
    AutoModel, 
    GPT2LMHeadModel, 
    GPT2Config
)
from .encoder import PROFA
from .classifier import MIXMLP
from .attention import RCTA

class CognitiveReportGenerator(nn.Module):
    """
    Complete model integrating all three modules:
    1. PRO-FA (Encoder)
    2. MIX-MLP (Classifier)
    3. RCTA (Attention Loop)
    + GPT-2 Decoder
    """
    def __init__(
        self,
        visual_encoder='vit_base_patch16_224',
        text_encoder_name='distilbert-base-uncased',
        decoder_name='distilgpt2',
        num_diseases=14,
        hidden_dim=512
    ):
        super().__init__()
        
        # Module 1: PRO-FA
        self.visual_encoder = PROFA(model_name=visual_encoder, pretrained=True)
        
        # Module 2: MIX-MLP
        self.disease_classifier = MIXMLP(in_dim=hidden_dim, num_diseases=num_diseases)
        
        # Module 3: RCTA
        self.triangular_attention = RCTA(d_model=hidden_dim, nhead=8)
        
        # Text encoder for clinical indication
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        text_hidden = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_hidden, hidden_dim)
        
        # Disease embedding (Projecting probs back to hidden dim for RCTA)
        self.disease_proj = nn.Linear(num_diseases, hidden_dim)
        
        # Report decoder
        self.decoder = GPT2LMHeadModel.from_pretrained(
            decoder_name,
            add_cross_attention=True
        )
        # Enable gradient checkpointing for memory efficiency
        self.decoder.gradient_checkpointing_enable()
        
        # Cross-attention adapter (to connect encoder output to decoder)
        decoder_hidden = self.decoder.config.n_embd
        if hidden_dim != decoder_hidden:
            self.adapter = nn.Linear(hidden_dim, decoder_hidden)
        else:
            self.adapter = nn.Identity()
        
        self.hidden_dim = hidden_dim
        
    def forward(
        self, 
        images, 
        indication_ids, 
        indication_mask, 
        report_ids=None,
        report_mask=None
    ):
        """
        Forward pass for training or inference preparation.
        """
        # 1. Extract visual features (PRO-FA)
        vis_feats = self.visual_encoder(images)
        organ_feat = vis_feats['organ']  # [B, 512]
        
        # 2. Predict diseases (MIX-MLP)
        disease_logits = self.disease_classifier(organ_feat)  # [B, 14]
        disease_probs = torch.sigmoid(disease_logits)
        
        # 3. Encode clinical indication
        # We process indication text to get a context vector
        text_outputs = self.text_encoder(
            input_ids=indication_ids,
            attention_mask=indication_mask
        )
        # Use [CLS] token representation
        clinical_feat = text_outputs.last_hidden_state[:, 0]
        clinical_feat = self.text_proj(clinical_feat)  # [B, 512]
        
        # 4. Apply Triangular Cognitive Attention (RCTA)
        # Project disease probs to embedding space
        disease_embed = self.disease_proj(disease_probs)  # [B, 512]
        
        # The core cognitive loop
        verified_feat = self.triangular_attention(
            img_feat=organ_feat,
            clinical_feat=clinical_feat,
            disease_feat=disease_embed
        )  # [B, 1, 512]
        
        # Adapt for decoder dimension
        encoder_hidden = self.adapter(verified_feat)  # [B, 1, decoder_hidden]
        
        # 5. Generate report or compute loss
        if report_ids is not None:
            # Training mode with labels
            outputs = self.decoder(
                input_ids=report_ids,
                attention_mask=report_mask,
                encoder_hidden_states=encoder_hidden,
                labels=report_ids,
                return_dict=True
            )
            
            return {
                'loss': outputs.loss,        # LM Cross Entropy Loss
                'disease_probs': disease_probs,
                'disease_logits': disease_logits
            }
        else:
            # Inference mode - return features needed for generation
            return {
                'encoder_hidden_states': encoder_hidden,
                'disease_probs': disease_probs,
                'disease_logits': disease_logits
            }
            
    def generate(self, images, indication_ids, indication_mask, max_length=256, num_beams=4):
        """
        Inference generation method
        """
        # Run forward to get encoder states
        # report_ids=None ensures we go to inference path in forward (though forward returns dict)
        features = self.forward(images, indication_ids, indication_mask)
        encoder_hidden = features['encoder_hidden_states']
        
        # Generate using GPT-2's generate method
        generated_ids = self.decoder.generate(
            encoder_hidden_states=encoder_hidden,
            encoder_attention_mask=torch.ones(encoder_hidden.shape[:2], device=encoder_hidden.device),
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            pad_token_id=self.decoder.config.eos_token_id 
        )
        
        return generated_ids, features['disease_probs']
