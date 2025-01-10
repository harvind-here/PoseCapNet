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


IMG_SIZE = 32
LATENT_DIM = 128
BATCH_SIZE = 8
NUM_EPOCHS = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_vocab(dataset, min_freq=5):
    counter = Counter()
    for _, caption, _ in dataset:
        counter.update(word_tokenize(caption.lower()))
    vocab = {word: idx for idx, (word, count) in enumerate(counter.items()) if count >= min_freq}
    vocab['<pad>'] = len(vocab)
    vocab['<unk>'] = len(vocab)
    return vocab, word_tokenize

def process_caption(caption, vocab, tokenizer, max_len=50):
    tokens = tokenizer(caption.lower())
    ids = [vocab.get(token, vocab['<unk>']) for token in tokens]
    ids = ids[:max_len]
    ids.extend([vocab['<pad>']] * (max_len - len(ids)))
    return torch.tensor(ids)


def train():
    dataset=COCODataset(img_dir='./mini_coco/val2017/', caption_file='./mini_coco/annotations/captions_val2017.json', keypoint_file = './mini_coco/annotations/person_keypoints_val2017.json')
    vocab, tokenizer = build_vocab(dataset)
    vocab_size = len(vocab)
    total_size = len(dataset)
    train_size = int(0.6*total_size)
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
    pose_criterion = nn.MSELoss()
    caption_criterion = nn.CrossEntropyLoss()
    ae_optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters()))
    task_optimizer = optim.Adam(list(pose_head.parameters())+list(caption_head.parameters()))

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        encoder.train()
        decoder.train()
        pose_head.train()
        caption_head.train()

        for batch_idx, (imgs, caption, keypoints) in enumerate(train_loader):
            imgs = imgs.to(DEVICE)
            keypoints = keypoints.to(DEVICE)
            caption = torch.stack([process_caption(cap, vocab, tokenizer) for cap in caption]).to(DEVICE)

            ae_optimizer.zero_grad()
            latent = encoder(imgs)
            reconstructed = decoder(latent)
            ae_loss = ae_criterion(reconstructed, imgs)
            ae_loss.backward()
            ae_optimizer.step()

            task_optimizer.zero_grad()
            with torch.no_grad():
                latent = encoder(imgs)
            pose_out = pose_head(latent)
            pose_loss = pose_criterion(pose_out, keypoints)
            caption_out = caption_head(latent)
            caption_loss = caption_criterion(caption_out.view(-1,vocab_size), caption.view(-1))
            total_loss = pose_loss+caption_loss
            total_loss.backward()
            task_optimizer.step()

            epoch_ae_loss += ae_loss.item()
            epoch_pose_loss += pose_loss.item()
            epoch_caption_loss += caption_loss.item()

            if batch_idx == len(train_loader)-1:
                epoch_time = int(time.time()-epoch_start_time)
                print(f"Epoch: {epoch}, Batch: {batch_idx}, "
                      f"AE Loss: {ae_loss.item():.4f}, "
                      f"Pose Loss: {pose_loss.item():.4f}, "
                      f"Caption Loss: {caption_loss.item():.4f}, "
                      f"Time: {epoch_time}s")
if __name__ == "__main__":
    train()