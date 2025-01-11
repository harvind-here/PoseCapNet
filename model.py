import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertTokenizer, BertModel, BertConfig

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Fix weights parameter
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        # Remove final avgpool and fc
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        # Add shared feature projection
        self.shared_conv = nn.Conv2d(512, 512, 1)
        
    def forward(self, x):
        # x: [B, 3, 224, 224]
        features = self.features(x)  # [B, 512, 7, 7]
        shared_features = self.shared_conv(features)  # [B, 512, 7, 7]
        return shared_features

class CaptionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab_size = self.tokenizer.vocab_size
        self.bert_config = BertConfig(
            vocab_size=self.vocab_size,
            hidden_size=768,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=2048
        )
        self.bert = BertModel(self.bert_config)
        self.spatial_proj = nn.Conv2d(512, 768, 1)
        # Add output projection
        self.output_proj = nn.Linear(768, self.vocab_size)
        
    def forward(self, features, captions=None):
        B = features.size(0)
        spatial_features = self.spatial_proj(features)  # [B, 768, 7, 7]
        sequence_features = spatial_features.flatten(2).transpose(1, 2)  # [B, 49, 768]
        
        if captions is not None:
            encoded = self.tokenizer(
                captions, 
                padding=True,
                truncation=True,
                max_length=49,
                return_tensors='pt'
            ).to(features.device)
            
            # Match sequence length to encoded input
            sequence_features = sequence_features[:, :encoded.input_ids.size(1), :]
            attention_mask = torch.ones((B, sequence_features.size(1)), device=features.device)
            
            bert_outputs = self.bert(
                inputs_embeds=sequence_features,
                attention_mask=attention_mask
            )
            logits = self.output_proj(bert_outputs.last_hidden_state)
            return logits
        else:
            bert_outputs = self.bert(inputs_embeds=sequence_features)
            return self.output_proj(bert_outputs.last_hidden_state)
        
    def get_vocab_size(self):
        return self.vocab_size

class HRNetPose(nn.Module):
    def __init__(self, num_joints=17):
        super().__init__()
        # Use spatial features directly
        self.conv1 = nn.Conv2d(512, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        
        self.high_res = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling
            nn.Flatten(),
            nn.Linear(64, num_joints * 3)  # Output x,y coordinates
        )
        
    def forward(self, x):
        # x: [B, 512, 7, 7]
        x = self.conv1(x)  # [B, 256, 7, 7]
        x = self.bn1(x)
        x = self.relu(x)
        return self.high_res(x)  # [B, num_joints * 2]