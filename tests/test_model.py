import sys
import unittest
from pathlib import Path
import torch

# Ensure UTF-8 output encoding for Windows compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.cognitive_model import CognitiveReportGenerator
from models.encoder import PROFA
from models.classifier import MIXMLP
from models.attention import RCTA


class TestCognitiveModel(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_diseases = 14
        self.hidden_dim = 512
        self.vocab_size = 30522
        
    def test_profa_encoder(self):
        print("\nTesting PRO-FA Encoder...")
        model = PROFA(model_name='vit_base_patch16_224', pretrained=False)
        images = torch.randn(self.batch_size, 3, 224, 224)
        outputs = model(images)
        
        self.assertEqual(outputs['organ'].shape, (self.batch_size, 512))
        self.assertEqual(outputs['pixel'].shape[0], self.batch_size)
        self.assertEqual(outputs['pixel'].shape[2], 512)
        print("[PASS] PRO-FA Encoder shapes correct")

    def test_mix_mlp(self):
        print("\nTesting MIX-MLP...")
        model = MIXMLP(in_dim=512, num_diseases=14)
        input_feat = torch.randn(self.batch_size, 512)
        logits = model(input_feat)
        
        self.assertEqual(logits.shape, (self.batch_size, 14))
        print("[PASS] MIX-MLP shapes correct")

    def test_rcta(self):
        print("\nTesting RCTA...")
        model = RCTA(d_model=512)
        img_feat = torch.randn(self.batch_size, 1, 512)
        clin_feat = torch.randn(self.batch_size, 1, 512)
        dis_feat = torch.randn(self.batch_size, 1, 512)
        
        output = model(img_feat, clin_feat, dis_feat)
        self.assertEqual(output.shape, (self.batch_size, 1, 512))
        print("[PASS] RCTA shapes correct")

    def test_full_model_forward(self):
        print("\nTesting Full Model Forward...")
        model = CognitiveReportGenerator(
            visual_encoder='vit_base_patch16_224',
            text_encoder_name='distilbert-base-uncased',
            decoder_name='distilgpt2',
            hidden_dim=512
        )
        
        images = torch.randn(self.batch_size, 3, 224, 224)
        indication_ids = torch.randint(0, 1000, (self.batch_size, 10))
        indication_mask = torch.ones(self.batch_size, 10)
        report_ids = torch.randint(0, 1000, (self.batch_size, 20))
        report_mask = torch.ones(self.batch_size, 20)
        
        output = model(
            images, 
            indication_ids, 
            indication_mask, 
            report_ids=report_ids, 
            report_mask=report_mask
        )
        
        self.assertIn('loss', output)
        self.assertIn('disease_logits', output)
        print("[PASS] Full Model Forward Pass successful")


if __name__ == '__main__':
    unittest.main()
