import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from model import ImageEncoder, CaptionDecoder, HRNetPose
from dataset import COCODataset, model_path
import time
from nltk.tokenize import word_tokenize
from collections import Counter
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


# Update configurations
IMG_SIZE = 224  # ResNet default size
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

def normalize_keypoints(keypoints, img_size):
    """Normalize keypoints to [0,1] range"""
    return keypoints / img_size

def train():
    # Dataset setup
    dataset = COCODataset(img_dir='./mini_coco/val2017/', 
                         caption_file='./mini_coco/annotations/captions_val2017.json',
                         keypoint_file='./mini_coco/annotations/person_keypoints_val2017.json')
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Initialize models
    encoder = ImageEncoder().to(DEVICE)
    caption_decoder = CaptionDecoder().to(DEVICE)
    pose_net = HRNetPose().to(DEVICE)
    
    # Vocab setup
    vocab_size = caption_decoder.get_vocab_size()
    
    # Loss functions
    caption_criterion = nn.CrossEntropyLoss(ignore_index=caption_decoder.tokenizer.pad_token_id)
    pose_criterion = nn.MSELoss()
    
    # Optimizers with different LRs
    encoder_optimizer = optim.AdamW(encoder.parameters(), lr=LEARNING_RATE)
    caption_optimizer = optim.AdamW(caption_decoder.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
    pose_optimizer = optim.AdamW(pose_net.parameters(), lr=LEARNING_RATE*2)

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        encoder.train()
        caption_decoder.train()
        pose_net.train()

        epoch_pose_loss = 0
        epoch_caption_loss = 0
        train_iter = tqdm(enumerate(train_loader), 
                         total=len(train_loader),
                         desc=f'Epoch [{epoch+1}/{NUM_EPOCHS}]')
        
        for batch_idx, (imgs, captions, keypoints) in train_iter:
            imgs = imgs.float()
            imgs = imgs.to(DEVICE)
            keypoints = keypoints.float()
            keypoints = keypoints/IMG_SIZE  # Normalize coordinates
            keypoints = keypoints.to(DEVICE)

            # Zero gradients
            encoder_optimizer.zero_grad()
            caption_optimizer.zero_grad()
            pose_optimizer.zero_grad()

            # Forward pass
            features = encoder(imgs)
            caption_output = caption_decoder(features, captions)
            encoded = caption_decoder.tokenizer(
                captions, 
                padding=True, 
                truncation=True,
                max_length=49,
                return_tensors='pt').to(features.device)
            pose_output = pose_net(features)


            # Debug shapes
            # B, S, V = caption_output.shape
            # print(f"Caption output shape: {caption_output.shape}")
            # print(f"Encoded input_ids shape: {encoded.input_ids.shape}")
            
            # Align batch sizes
            caption_output = caption_output[:, :encoded.input_ids.size(1), :]  # Match sequence length
                
            # Calculate losses
            caption_loss = caption_criterion(caption_output.reshape(-1, vocab_size), encoded.input_ids.reshape(-1))
            pose_loss = pose_criterion(pose_output, keypoints)
            total_loss = caption_loss + pose_loss
            total_loss.backward()

            # Update weights
            encoder_optimizer.step()
            caption_optimizer.step()
            pose_optimizer.step()

            # Progress tracking
            train_iter.set_postfix({
                'pose': f'{pose_loss.item():.4f}',
                'cap': f'{caption_loss.item():.4f}'
            })

            epoch_pose_loss += pose_loss.item()
            epoch_caption_loss += caption_loss.item()

        # Calculate averages
        avg_pose_loss = epoch_pose_loss / len(train_loader)
        avg_caption_loss = epoch_caption_loss / len(train_loader)
        epoch_time = int(time.time() - epoch_start_time)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} completed in {epoch_time}s")
        print(f"Avg losses - Pose: {avg_pose_loss:.4f}, Caption: {avg_caption_loss:.4f}\n")
        
        # Early stopping and checkpointing
        val_loss = avg_pose_loss + avg_caption_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'encoder': encoder.state_dict(),
                'caption_decoder': caption_decoder.state_dict(),
                'pose_net': pose_net.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, model_path())
            print(f"Saved best model with val_loss: {val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

if __name__ == "__main__":
    train()