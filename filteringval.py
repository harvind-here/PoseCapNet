import json
import os
from pathlib import Path

def filter_instance_annotations(data, image_ids):
    return {
        'info': data['info'],
        'licenses': data['licenses'],
        'categories': data['categories'],
        'images': [img for img in data['images'] if img['id'] in image_ids],
        'annotations': [ann for ann in data['annotations'] if ann['image_id'] in image_ids]
    }

def filter_caption_annotations(data, image_ids):
    return {
        'info': data['info'],
        'licenses': data['licenses'],
        'images': [img for img in data['images'] if img['id'] in image_ids],
        'annotations': [ann for ann in data['annotations'] if ann['image_id'] in image_ids]
    }

def filter_annotations(annotation_file, image_dir):
    # Load original annotations
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Get list of available image files
    available_images = set(int(f.stem) for f in Path(image_dir).glob('*.jpg'))
    
    # Filter images and get image IDs
    filtered_images = [img for img in data['images'] 
                      if int(img['file_name'].split('.')[0]) in available_images]
    image_ids = set(img['id'] for img in filtered_images)
    
    # Determine annotation type and apply appropriate filtering
    if 'categories' in data:
        return filter_instance_annotations(data, image_ids)
    else:
        return filter_caption_annotations(data, image_ids)

def main():
    base_dir = os.path.join(os.getcwd(), 'mini_coco')
    val_img_dir = os.path.join(base_dir, 'val2017')
    ann_dir = os.path.join(base_dir, 'annotations')
    
    files = [
        'instances_val2017.json',
        'person_keypoints_val2017.json',
        'captions_val2017.json'
    ]
    
    for file in files:
        try:
            print(f"Processing {file}...")
            input_path = os.path.join(ann_dir, file)
            output_path = os.path.join(ann_dir, f'filtered_{file}')
            
            if os.path.exists(input_path):
                filtered_data = filter_annotations(input_path, val_img_dir)
                
                with open(output_path, 'w') as f:
                    json.dump(filtered_data, f)
                print(f"Saved filtered annotations to {output_path}")
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")

if __name__ == "__main__":
    main()