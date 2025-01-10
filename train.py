import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from model import Encoder, Decoder, PoseHead, CaptionHead
from dataset import COCODataset
import time
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from collections import Counter
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from torchvision.transforms.functional import to_pil_image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

IMG_SIZE = 128
LATENT_DIM = 512
BATCH_SIZE = 64
NUM_EPOCHS = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

def calculate_metrics(model_dict, data_loader, vocab, device):
    encoder = model_dict['encoder'].eval()
    decoder = model_dict['decoder'].eval()
    pose_head = model_dict['pose_head'].eval()
    caption_head = model_dict['caption_head'].eval()
    
    pose_mse = 0
    recon_psnr = 0
    recon_ssim = 0
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for imgs, captions, keypoints in data_loader:
            imgs = imgs.to(device).float()
            keypoints = normalize_keypoints(keypoints.to(device).float(), IMG_SIZE)
            
            latent = encoder(imgs)
            reconstructed = decoder(latent)
            pose_out = pose_head(latent)
            caption_out = caption_head(latent)
            
            pose_mse += nn.functional.mse_loss(pose_out, keypoints).item()
            
            # Modified image metrics calculation
            for img, rec in zip(imgs.cpu(), reconstructed.cpu()):
                img_np = np.array(to_pil_image(img))
                rec_np = np.array(to_pil_image(rec))
                
                try:
                    recon_psnr += psnr(img_np, rec_np)
                    recon_ssim += ssim(img_np, rec_np, 
                                     win_size=3,  # Smaller window size
                                     channel_axis=2,  # RGB channel axis
                                     data_range=255)
                except ValueError as e:
                    print(f"Warning: SSIM calculation failed - {e}")
                    continue
            
             # Fix caption metrics calculation
            caption_ids = caption_out.argmax(dim=-1)
            for ref, hyp in zip(captions, caption_ids):
                references.append([ref.split()])
                # Convert indices to words using vocab dict
                hypothesis = []
                for idx in hyp:
                    idx_item = idx.item()
                    # Find word by index in vocab
                    for word, vidx in vocab.items():
                        if vidx == idx_item:
                            hypothesis.append(word)
                            break
                hypotheses.append(hypothesis)
    
    
    num_samples = len(data_loader.dataset)
    metrics = {
        'pose_mse': pose_mse / len(data_loader),
        'recon_psnr': recon_psnr / num_samples,
        'recon_ssim': recon_ssim / num_samples,
        'bleu1': corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0)),
        'bleu4': corpus_bleu(references, hypotheses)
    }
    
    return metrics

def build_vocab(dataset, min_freq=5):
    counter = Counter()
    for _, caption, _ in dataset:
        counter.update(word_tokenize(caption.lower()))
     # Start with special tokens
    vocab = {
        '<pad>': 0,
        '<unk>': 1,
        '<start>': 2,
        '<end>': 3
    }
    
    # Add words that meet minimum frequency
    idx = len(vocab)
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab, word_tokenize

def process_caption(caption, vocab, tokenizer, max_len=50):
    tokens = tokenizer(caption.lower())
    # Reserve space for start and end tokens
    max_sequence_len = max_len - 2
    
    # Convert tokens to indices with bounds checking
    indices = []
    indices.append(vocab['<start>'])  # Add start token
    
    # Add word tokens
    for token in tokens[:max_sequence_len]:
        if token in vocab:
            indices.append(vocab[token])
        else:
            indices.append(vocab['<unk>'])
    
    indices.append(vocab['<end>'])  # Add end token
    
    # Pad to max_len
    padding_length = max_len - len(indices)
    if padding_length > 0:
        indices.extend([vocab['<pad>']] * padding_length)
    
    # Verify all indices are within bounds
    vocab_size = len(vocab)
    for idx in indices:
        assert 0 <= idx < vocab_size, f"Token index {idx} out of bounds (vocab size: {vocab_size})"
    
    return torch.tensor(indices)

def normalize_keypoints(keypoints, img_size):
    """Normalize keypoints to [0,1] range"""
    return keypoints / img_size

def train():
    dataset=COCODataset(img_dir='./mini_coco/val2017/', caption_file='./mini_coco/annotations/captions_val2017.json', keypoint_file = './mini_coco/annotations/person_keypoints_val2017.json')
    vocab, tokenizer = build_vocab(dataset)
    vocab_size = len(vocab)
    total_size = len(dataset)
    train_size = int(0.7*total_size)
    val_size = int(0.2*total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    encoder = Encoder(LATENT_DIM).to(DEVICE)
    decoder = Decoder(LATENT_DIM).to(DEVICE)
    pose_head = PoseHead(LATENT_DIM).to(DEVICE)
    caption_head = CaptionHead(LATENT_DIM, vocab_size).to(DEVICE)
    
    ae_criterion = nn.MSELoss()
    pose_criterion = nn.MSELoss(reduction='mean')
    caption_criterion = nn.CrossEntropyLoss()
    ae_optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters()))
    task_optimizer = optim.Adam(list(pose_head.parameters())+list(caption_head.parameters()))

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        encoder.train()
        decoder.train()
        pose_head.train()
        caption_head.train()

        epoch_ae_loss = 0
        epoch_pose_loss = 0
        epoch_caption_loss = 0

        # Initialize progress bar
        train_iter = tqdm(enumerate(train_loader), 
                         total=len(train_loader),
                         desc=f'Epoch [{epoch+1}/{NUM_EPOCHS}]',
                         leave=True)
        
        for batch_idx, (imgs, caption, keypoints) in train_iter:
            imgs = imgs.to(DEVICE).float()
            keypoints = normalize_keypoints(keypoints.to(DEVICE).float(), IMG_SIZE)           
            caption_tokens = torch.stack([process_caption(cap, vocab, tokenizer) for cap in caption]).to(DEVICE).long()
            batch_size = imgs.size(0)
            # print(f"Batch size: {batch_size}")
            # print(f"Caption tokens shape: {caption_tokens.shape}")

            ae_optimizer.zero_grad()
            latent = encoder(imgs)
            reconstructed = decoder(latent)
            ae_loss = ae_criterion(reconstructed, imgs)
            ae_loss.backward()
            ae_optimizer.step()

            task_optimizer.zero_grad()
            with torch.no_grad():
                latent = encoder(imgs)
            pose_out = pose_head(latent).float()
            pose_loss = pose_criterion(pose_out, keypoints)*0.01 
            
            caption_out = caption_head(latent)
            # print(f"Caption output shape: {caption_out.shape}")
             # Reshape caption outputs for loss calculation
            caption_out = caption_out.view(batch_size, -1, vocab_size)  # [batch_size, max_len, vocab_size]
            caption_tokens = caption_tokens.view(batch_size, -1)        # [batch_size, max_len]
            # More debug prints
            # print(f"Reshaped caption output: {caption_out.shape}")
            # print(f"Reshaped caption tokens: {caption_tokens.shape}")
            
            caption_loss = caption_criterion(caption_out.view(-1, vocab_size),caption_tokens.view(-1))
            total_loss = pose_loss+caption_loss
            total_loss.backward()
            task_optimizer.step()

            # Update progress bar postfix
            train_iter.set_postfix({
                'ae': f'{ae_loss.item():.4f}',
                'pose': f'{pose_loss.item():.4f}',
                'cap': f'{caption_loss.item():.4f}'
            }, refresh=True)

            epoch_ae_loss += ae_loss.item()
            epoch_pose_loss += pose_loss.item()
            epoch_caption_loss += caption_loss.item()
        
        train_iter.close()
        # Print epoch summary
        avg_ae_loss = epoch_ae_loss / len(train_loader)
        avg_pose_loss = epoch_pose_loss / len(train_loader)
        avg_caption_loss = epoch_caption_loss / len(train_loader)
        epoch_time = int(time.time() - epoch_start_time)
        
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} completed in {epoch_time}s")
        print(f"Avg losses - AE: {avg_ae_loss:.4f}, Pose: {avg_pose_loss:.4f}, Caption: {avg_caption_loss:.4f}\n")
        
        # Evaluate on validation set
        val_metrics = calculate_metrics(
            {'encoder': encoder, 'decoder': decoder, 
             'pose_head': pose_head, 'caption_head': caption_head},
            val_loader, vocab, DEVICE
        )
        
        print("\n\nValidation Metrics:")
        print(f"Pose MSE: {val_metrics['pose_mse']:.4f}")
        print(f"Reconstruction - PSNR: {val_metrics['recon_psnr']:.2f}, SSIM: {val_metrics['recon_ssim']:.4f}")
        print(f"Caption - BLEU-1: {val_metrics['bleu1']:.4f}, BLEU-4: {val_metrics['bleu4']:.4f}")
        
            # if batch_idx == len(train_loader)-1:
            #     epoch_time = int(time.time()-epoch_start_time)
            #     print(f"Epoch: {epoch}, Batch: {batch_idx}, "
            #           f"AE Loss: {ae_loss.item():.4f}, "
            #           f"Pose Loss: {pose_loss.item():.4f}, "
            #           f"Caption Loss: {caption_loss.item():.4f}, "
            #           f"Time: {epoch_time}s")

if __name__ == "__main__":
    train()