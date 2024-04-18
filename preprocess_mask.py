import os
import glob
from PIL import Image
import click
from tqdm import tqdm
from rembg import remove

@click.command()
@click.option('--data', required=True, help='Input directory')  
def predict(data):
    input_dir, output_dir = os.path.join(data, 'images'), os.path.join(data, 'masks')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image paths
    image_paths = glob.glob(os.path.join(input_dir, '*.jpg')) + glob.glob(os.path.join(input_dir, '*.png')) + glob.glob(os.path.join(input_dir, '*.JPG'))
    
    for img_path in tqdm(sorted(image_paths)):# Load and preprocess image
        img = Image.open(img_path)
        
        mask_img = remove(img, only_mask=True)
        
        # Save output
        filename = os.path.basename(img_path)
        mask_img.save(os.path.join(output_dir, filename))
        
if __name__ == '__main__':
    predict()