import torch
from torch.utils.data import DataLoader
from model import ImageEncoder, CaptionDecoder, HRNetPose
from dataset import COCODataset, model_path
from tqdm import tqdm
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import json
import os
import numpy as np

def plot_skeleton(img, keypoints):
    """
    Plot skeleton on the image using predicted keypoints
    keypoints: shape [51] -> reshape to [17, 3] for x,y,confidence
    """
    # Define skeleton connections (COCO format)
    skeleton = [
        [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13], #arms
        [6,7],[6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3], #body
        [2,4],[3,5],[4,6],[5,7] #face
    ]
    
    # Colors for visualization
    colors = plt.cm.rainbow(np.linspace(0, 1, len(skeleton)))
    
    plt.figure(figsize=(8, 8))
    
    # Show image
    img_np = img.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    plt.imshow(img_np)
    
    # Reshape keypoints to [17, 3]
    keypoints = keypoints.reshape(-1, 3)
    keypoints[:, :2] *= 224  # Denormalize coordinates
    
    # Plot joints
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c='red', s=30)
    
    # Plot skeleton
    for i, (j1, j2) in enumerate(skeleton):
        if keypoints[j1-1, 2] > 0 and keypoints[j2-1, 2] > 0:  # Check visibility
            plt.plot([keypoints[j1-1, 0], keypoints[j2-1, 0]],
                    [keypoints[j1-1, 1], keypoints[j2-1, 1]],
                    color=colors[i], linewidth=2)
    
    plt.axis('off')
    return plt.gcf()

def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    encoder = ImageEncoder().to(device)
    caption_decoder = CaptionDecoder().to(device)
    pose_net = HRNetPose().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path())
    encoder.load_state_dict(checkpoint['encoder'])
    caption_decoder.load_state_dict(checkpoint['caption_decoder'])
    pose_net.load_state_dict(checkpoint['pose_net'])
    
    # Set to eval mode
    encoder.eval()
    caption_decoder.eval()
    pose_net.eval()
    
    # Get test dataset
    dataset = COCODataset(
        img_dir='./mini_coco/val2017/',
        caption_file='./mini_coco/annotations/captions_val2017.json',
        keypoint_file='./mini_coco/annotations/person_keypoints_val2017.json'
    )
    
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    _, _, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Evaluation metrics
    pose_mse = 0
    references = []
    hypotheses = []
    results = []
    
    with torch.no_grad():
        for batch_idx, (imgs, captions, keypoints) in enumerate(tqdm(test_loader, desc="Testing")):
            imgs = imgs.to(device)
            keypoints = keypoints.to(device).float() / 224  # Normalize
            
            # Forward pass
            features = encoder(imgs)
            pose_outputs = pose_net(features)
            caption_outputs = caption_decoder(features)
            
            # Calculate metrics
            pose_mse += torch.nn.functional.mse_loss(pose_outputs, keypoints.reshape(keypoints.size(0), -1)).item()
            
            # Visualize each image in batch
            for i in range(len(imgs)):
                # Visualization code remains same
                fig = plot_skeleton(imgs[i], pose_outputs[i].cpu())
                save_dir = os.path.join(os.path.dirname(model_path()), 'pose_visualizations')
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir, f'pose_{batch_idx}_{i}.png'))
                plt.close(fig)
                
                # Updated caption processing
                ref_caption = captions[i]
                pred_tokens = caption_outputs[i].argmax(dim=-1)
                pred_caption = caption_decoder.tokenizer.decode(pred_tokens, skip_special_tokens=True)
                
                # Add to references and hypotheses for BLEU
                references.append([ref_caption.split()])
                hypotheses.append(pred_caption.split())
                
                results.append({
                    'reference': ref_caption,
                    'prediction': pred_caption,
                    'pose_error': pose_outputs[i].cpu().numpy().tolist(),
                    'visualization_path': os.path.join(save_dir, f'pose_{batch_idx}_{i}.png')
                })

    # Calculate final metrics
    avg_pose_mse = pose_mse / len(test_loader)
    smooth = SmoothingFunction().method1
    
    # Safe BLEU calculation
    if len(references) > 0 and len(hypotheses) > 0:
        bleu1 = corpus_bleu(references, hypotheses, 
                           weights=(1.0, 0, 0, 0),
                           smoothing_function=smooth)
        bleu4 = corpus_bleu(references, hypotheses,
                           smoothing_function=smooth)
    else:
        bleu1 = bleu4 = 0.0
    
    print(f"\nTest Results:")
    print(f"Pose MSE: {avg_pose_mse:.4f}")
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")
    
    # Save results
    results_file = os.path.join(os.path.dirname(model_path()), 'test_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'metrics': {
                'pose_mse': avg_pose_mse,
                'bleu1': bleu1,
                'bleu4': bleu4
            },
            'predictions': results
        }, f, indent=2)
    
    print(f"\nResults saved to {results_file}")

if __name__ == "__main__":
    test()