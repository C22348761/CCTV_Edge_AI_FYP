#!/usr/bin/env python3
"""
Batch background removal script for surveillance images.
Processes all images in Real_bg_before/day and night folders, removes people/objects,
and saves clean backgrounds to Real_bg_after with day_bg_001, night_bg_001 naming.
"""

from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import os
from pathlib import Path
from dotenv import load_dotenv
import time

# Load environment variables from .env file
load_dotenv()

# Create client (API key should be set as environment variable)
client = genai.Client()

# Enhanced prompt for removing people while keeping scene identical
prompt = (
    "Remove the detected person from this surveillance image while keeping the background scene "
    "completely identical. Remove all bounding boxes, detection labels, and text overlays. "
    "The result should: "
    "- Remove the person entirely from the frame "
    "- Keep the background environment exactly the same "
    "- Maintain the same lighting, shadows, and atmospheric conditions "
    "- Preserve all environmental elements (buildings, objects, terrain, etc.) "
    "- Fill the area where the person was with appropriate background elements "
    "- Keep the same surveillance camera quality and characteristics "
    "- Ensure the scene looks natural and unaltered except for the person removal "
    "- Maintain consistent perspective and camera angle "
    "- No visible signs of editing or manipulation"
)

def create_output_folders(base_path):
    """Create the Real_bg_after folder structure."""
    bg_after_path = base_path / "Real_bg_after"
    day_path = bg_after_path / "day"
    night_path = bg_after_path / "night"
    
    # Create directories if they don't exist
    day_path.mkdir(parents=True, exist_ok=True)
    night_path.mkdir(parents=True, exist_ok=True)
    
    return day_path, night_path

def process_image(image_path, output_path, prefix, image_id):
    """Process a single image through the background removal pipeline."""
    try:
        print(f"  Processing {image_path.name}...")
        
        # Load the input image
        image = Image.open(image_path)
        
        # Generate content using both text prompt and image
        response = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=[prompt, image],
        )
        
        # Process the response
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                # Save the generated clean background image
                cleaned_image = Image.open(BytesIO(part.inline_data.data))
                output_filename = f"{prefix}_bg_{image_id:03d}.jpg"
                output_filepath = output_path / output_filename
                cleaned_image.save(output_filepath)
                print(f"    ✅ Saved: {output_filename}")
                return True
        
        print(f"    ❌ No image data returned for {image_path.name}")
        return False
        
    except Exception as e:
        print(f"    ❌ Error processing {image_path.name}: {e}")
        return False

def process_folder(folder_path, output_path, prefix):
    """Process all images in a folder."""
    if not folder_path.exists():
        print(f"❌ Input folder {folder_path} does not exist!")
        return 0
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = [f for f in folder_path.iterdir() if f.suffix in image_extensions and f.is_file()]
    
    if not image_files:
        print(f"❌ No images found in {folder_path}")
        return 0
    
    print(f"\n🔄 Processing {len(image_files)} images from {folder_path.name}/")
    print(f"📁 Output folder: {output_path}")
    
    # Sort files for consistent processing
    image_files.sort(key=lambda x: x.name)
    
    successful_count = 0
    for i, image_file in enumerate(image_files, 1):
        success = process_image(image_file, output_path, prefix, i)
        if success:
            successful_count += 1
        
        # Add a small delay to avoid rate limiting
        time.sleep(1)
    
    print(f"✅ Successfully processed {successful_count}/{len(image_files)} images from {folder_path.name}/")
    return successful_count

def main():
    """Main function to process all surveillance images."""
    base_path = Path("/Users/tadhgroche/Documents/Data-FYP /CCTV_DATA_FYP/Real_Images")
    
    print("🔄 Starting batch background removal for surveillance images...")
    print("=" * 70)
    
    # Create output folder structure
    day_output, night_output = create_output_folders(base_path)
    
    # Process day images
    day_input = base_path / "Real_bg_before" / "day"
    day_count = process_folder(day_input, day_output, "day")
    
    # Process night images
    night_input = base_path / "Real_bg_before" / "night"
    night_count = process_folder(night_input, night_output, "night")
    
    print("\n" + "=" * 70)
    print("✅ Batch background removal completed!")
    print(f"📊 Results:")
    print(f"   Day images processed: {day_count}")
    print(f"   Night images processed: {night_count}")
    print(f"   Total processed: {day_count + night_count}")
    print(f"\n📁 Clean backgrounds saved to:")
    print(f"   Day: {day_output}")
    print(f"   Night: {night_output}")
    print(f"\n📝 Naming scheme: day_bg_001.jpg, day_bg_002.jpg, etc.")

if __name__ == "__main__":
    main()
