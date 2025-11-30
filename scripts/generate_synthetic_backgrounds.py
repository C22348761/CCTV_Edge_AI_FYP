#!/usr/bin/env python3
"""
Synthetic CCTV Background Generator using Google Nano Banana
Generates 10 synthetic backgrounds matching the exact style and setting of real CCTV images.
Creates both day (outdoor residential) and night (outdoor residential) backgrounds with varied angles and areas.
"""

from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
import time

class SyntheticCCTVGenerator:
    def __init__(self, output_dir, reference_day_path, reference_night_path):
        self.output_dir = output_dir
        self.reference_day_path = reference_day_path
        self.reference_night_path = reference_night_path
        
        # Load environment variables
        load_dotenv()
        
        # Create client
        self.client = genai.Client()
        
        # Create output directories
        os.makedirs(os.path.join(output_dir, "day"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "night"), exist_ok=True)
        
    def get_day_prompt(self, variation):
        """Get day-specific prompt for outdoor residential scenes with varied angles"""
        day_scenes = [
            # Images 1-29: Residential house-mounted CCTV looking outward - International styles
            "American single-story suburban backyard camera from house looking out at deck and pool area",
            "Swedish two-story house backyard camera looking out at wooden deck and birch trees",
            "British single-story terraced house backyard camera looking out at small garden and fence",
            "American single-story ranch house backyard camera looking out at patio and lawn",
            "German two-story house backyard camera looking out at garden and hedge",
            "Australian single-story house backyard camera looking out at veranda and native plants",
            "Canadian two-story house backyard camera looking out at deck and maple trees",
            "Dutch single-story house backyard camera looking out at garden and canal view",
            "American two-story colonial house backyard camera looking out at porch and garden",
            "Scandinavian single-story house backyard camera looking out at deck and pine trees",
            "American two-story suburban house side entrance camera looking out at driveway and lawn",
            "British single-story Victorian house front camera looking out at garden and street",
            "American single-story ranch house garage camera looking out at driveway and street",
            "Swedish two-story house front door camera looking out at walkway and birch trees",
            "German single-story house corner camera looking out at front yard and hedge",
            "Australian two-story house wall camera looking out at backyard and native garden",
            "American single-story colonial house entrance camera looking out at steps and landscaping",
            "Dutch two-story house side camera looking out at walkway and canal",
            "Canadian single-story house front porch camera looking out at driveway and maple trees",
            "British two-story house garden camera looking out at flower beds and fence",
            "American single-story suburban house wall camera looking out at side yard and lawn",
            "Scandinavian two-story house front entrance camera looking out at walkway and pine trees",
            "German single-story house garage camera looking out at driveway and hedge",
            "Australian two-story house side camera looking out at entrance and native plants",
            "American single-story ranch house corner camera looking out at front yard and driveway",
            "Swedish two-story house garden camera looking out at flower beds and birch trees",
            "British single-story house front porch camera looking out at steps and street",
            "Dutch two-story house corner camera looking out at garden and canal view",
            "Canadian single-story house wall camera looking out at side yard and maple trees",
            
            # Images 30-39: Countryside scenes
            "countryside farmhouse with barn and fields",
            "rural property with fence and open fields",
            "countryside driveway with trees and hedges",
            "farm entrance with gate and rural landscape",
            "countryside garden with vegetable patches",
            "rural backyard with chicken coop and fields",
            "countryside patio overlooking farmland",
            "rural walkway with wildflowers and meadows",
            "countryside garage with tractor and farm equipment",
            "rural property corner showing barn and pastures",
            
            # Images 40-49: Commercial/Retail - Building-mounted CCTV looking outward
            "petrol station building wall camera mounted at 3 meters looking out at forecourt and fuel pumps",
            "city centre shop building wall camera mounted at entrance level looking out at street and pedestrians",
            "shopping centre building wall camera mounted at 2.5 meters looking out at parking area and entrance",
            "retail store building wall camera mounted at shop level looking out at parking area and trolleys",
            "convenience store building wall camera mounted at entrance looking out at outdoor seating and street",
            "petrol station building corner camera mounted at 3 meters looking out at forecourt and pumps",
            "city centre building wall camera mounted at shop level looking out at pedestrian walkway and street",
            "retail complex building wall camera mounted at entrance level looking out at multiple shop entrances",
            "petrol station building wall camera mounted at 2.5 meters looking out at forecourt and car wash",
            "city centre shop building wall camera mounted at shop level looking out at street and shop fronts"
        ]
        
        scene = day_scenes[variation % len(day_scenes)]
        
        return (
            f"Create a SINGLE surveillance camera background scene that matches the exact same style and quality "
            f"as the reference image. The new background must have: "
            f"- EXACT same surveillance camera quality (slightly blurry, not high-resolution, low-quality) "
            f"- EXACT same color grading and desaturation "
            f"- EXACT same lighting characteristics (soft, diffused lighting) "
            f"- EXACT same surveillance camera angle and perspective (elevated, looking down, but NOT bird's eye view) "
            f"- CRITICAL: Camera mounting height based on building type: "
            f"  * Single-story house: Mount camera HIGH (3-4 meters) looking down at ground level "
            f"  * Two-story house: Mount camera LOWER (2-2.5 meters) looking slightly down "
            f"  * Commercial buildings: Mount at entrance level (2.5-3 meters) looking outward "
            f"- CRITICAL: NO bird's eye view or roof-mounted cameras - show realistic CCTV mounting height "
            f"- CRITICAL: Camera should NOT appear to be on the roof of the house "
            f"- EXACT same grain, texture, and compression artifacts as the reference image "
            f"- EXACT same surveillance camera distortion and field of view "
            f"- EXACT same image quality degradation, noise, and pixelation as the reference "
            f"- EXACT same compression artifacts and digital noise patterns "
            f"- EXACT same low-resolution, grainy surveillance camera aesthetic "
            f"- Slightly more grainy and pixelated than the reference - like older, lower-end surveillance camera "
            f"- Slightly more compression artifacts and digital noise for authentic CCTV look "
            f"- EXACT same timestamp overlay style (top-right corner, white digits, fuzzy) "
            f"- EXACT same camera label style (bottom-left corner, faint, low-res) "
            f"- Create a DIFFERENT household outdoor environment but maintain the exact same surveillance aesthetic "
            f"- The new scene should be a {scene} "
            f"- Focus on outdoor areas that would be monitored by surveillance cameras "
            f"- CRITICAL: This is a RESIDENTIAL scene - show typical suburban/urban house and garden areas "
            f"- CRITICAL: NO countryside, rural, farm, or agricultural elements - this is a residential house "
            f"- CRITICAL: Show typical residential elements: house walls, driveways, gardens, fences, walkways "
            f"- Vary the camera positioning: some from corner angles, some from side views, some from different heights "
            f"- Show different perspectives: corner views, side angles, entrance views, garden areas "
            f"- Include realistic camera mounting positions: house corners, wall-mounted, roof-mounted, entrance-mounted "
            f"- Show cameras looking outward from the HOUSE towards garden, driveway, street, and walkways "
            f"- Include corner cameras that capture multiple areas simultaneously "
            f"- Include residential outdoor elements: house walls, driveways, gardens, fences, walkways, patios "
            f"- Show typical suburban residential setting with house, garden, driveway, and street access "
            f"- NO fisheye lens distortion - maintain the exact same camera lens type as the reference "
            f"- Keep the same field of view and perspective style as the reference image "
            f"- Ensure the camera angle and lens characteristics match exactly "
            f"- Ensure the new background looks like it was captured by the same surveillance camera system "
            f"- Maintain the same vintage, slightly aged surveillance footage look "
            f"- Keep the same overall mood and atmosphere as the reference "
            f"- Make it look like authentic surveillance footage from the same camera system "
            f"- CRITICAL: Generate ONLY ONE single image, NOT multiple images or a grid of images "
            f"- CRITICAL: NO humans, people, or persons anywhere in the scene - this is a background only "
            f"- CRITICAL: NO vehicles with people inside - only empty cars or no cars at all "
            f"- CRITICAL: Match the EXACT same low-quality, grainy, pixelated look as the reference image "
            f"- CRITICAL: Do NOT generate high-quality, sharp, or clean images - they must look like old CCTV footage "
            f"- The only difference should be the physical location/scene and camera angle, everything else should match exactly"
        )
    
    def get_night_prompt(self, variation):
        """Get night-specific prompt for outdoor residential scenes with varied angles"""
        night_scenes = [
            # Images 1-29: Residential house-mounted CCTV looking outward at night - International styles
            "American single-story suburban backyard camera from house looking out at deck and pool area at night",
            "Swedish two-story house backyard camera looking out at wooden deck and birch trees at night",
            "British single-story terraced house backyard camera looking out at small garden and fence at night",
            "American single-story ranch house backyard camera looking out at patio and lawn at night",
            "German two-story house backyard camera looking out at garden and hedge at night",
            "Australian single-story house backyard camera looking out at veranda and native plants at night",
            "Canadian two-story house backyard camera looking out at deck and maple trees at night",
            "Dutch single-story house backyard camera looking out at garden and canal view at night",
            "American two-story colonial house backyard camera looking out at porch and garden at night",
            "Scandinavian single-story house backyard camera looking out at deck and pine trees at night",
            "American two-story suburban house side entrance camera looking out at driveway and lawn at night",
            "British single-story Victorian house front camera looking out at garden and street at night",
            "American single-story ranch house garage camera looking out at driveway and street at night",
            "Swedish two-story house front door camera looking out at walkway and birch trees at night",
            "German single-story house corner camera looking out at front yard and hedge at night",
            "Australian two-story house wall camera looking out at backyard and native garden at night",
            "American single-story colonial house entrance camera looking out at steps and landscaping at night",
            "Dutch two-story house side camera looking out at walkway and canal at night",
            "Canadian single-story house front porch camera looking out at driveway and maple trees at night",
            "British two-story house garden camera looking out at flower beds and fence at night",
            "American single-story suburban house wall camera looking out at side yard and lawn at night",
            "Scandinavian two-story house front entrance camera looking out at walkway and pine trees at night",
            "German single-story house garage camera looking out at driveway and hedge at night",
            "Australian two-story house side camera looking out at entrance and native plants at night",
            "American single-story ranch house corner camera looking out at front yard and driveway at night",
            "Swedish two-story house garden camera looking out at flower beds and birch trees at night",
            "British single-story house front porch camera looking out at steps and street at night",
            "Dutch two-story house corner camera looking out at garden and canal view at night",
            "Canadian single-story house wall camera looking out at side yard and maple trees at night",
            
            # Images 30-39: Countryside scenes
            "countryside farmhouse with barn and fields at night",
            "rural property with fence and open fields at night",
            "countryside driveway with trees and hedges at night",
            "farm entrance with gate and rural landscape at night",
            "countryside garden with vegetable patches at night",
            "rural backyard with chicken coop and fields at night",
            "countryside patio overlooking farmland at night",
            "rural walkway with wildflowers and meadows at night",
            "countryside garage with tractor and farm equipment at night",
            "rural property corner showing barn and pastures at night",
            
            # Images 40-49: Commercial/Retail - Building-mounted CCTV looking outward
            "petrol station building wall camera mounted at 3 meters looking out at forecourt and fuel pumps at night",
            "city centre shop building wall camera mounted at entrance level looking out at street and pedestrians at night",
            "shopping centre building wall camera mounted at 2.5 meters looking out at parking area and entrance at night",
            "retail store building wall camera mounted at shop level looking out at parking area and trolleys at night",
            "convenience store building wall camera mounted at entrance looking out at outdoor seating and street at night",
            "petrol station building corner camera mounted at 3 meters looking out at forecourt and pumps at night",
            "city centre building wall camera mounted at shop level looking out at pedestrian walkway and street at night",
            "retail complex building wall camera mounted at entrance level looking out at multiple shop entrances at night",
            "petrol station building wall camera mounted at 2.5 meters looking out at forecourt and car wash at night",
            "city centre shop building wall camera mounted at shop level looking out at street and shop fronts at night"
        ]
        
        scene = night_scenes[variation % len(night_scenes)]
        
        return (
            f"Create a SINGLE surveillance camera background scene that matches the exact same style and quality "
            f"as the reference image. The new background must have: "
            f"- EXACT same surveillance camera quality (slightly blurry, not high-resolution, low-quality) "
            f"- EXACT same color grading and desaturation (grayscale/monochrome) "
            f"- EXACT same lighting characteristics (dim, low-light, infrared-style) "
            f"- EXACT same surveillance camera angle and perspective (elevated, looking down, but NOT bird's eye view) "
            f"- CRITICAL: Camera mounting height based on building type: "
            f"  * Single-story house: Mount camera HIGH (3-4 meters) looking down at ground level "
            f"  * Two-story house: Mount camera LOWER (2-2.5 meters) looking slightly down "
            f"  * Commercial buildings: Mount at entrance level (2.5-3 meters) looking outward "
            f"- CRITICAL: NO bird's eye view or roof-mounted cameras - show realistic CCTV mounting height "
            f"- CRITICAL: Camera should NOT appear to be on the roof of the house "
            f"- EXACT same grain, texture, and compression artifacts as the reference image "
            f"- EXACT same surveillance camera distortion and field of view "
            f"- EXACT same image quality degradation, noise, and pixelation as the reference "
            f"- EXACT same compression artifacts and digital noise patterns "
            f"- EXACT same low-resolution, grainy surveillance camera aesthetic "
            f"- EVEN LOWER quality than day scenes - more grainy, more pixelated, more compression artifacts "
            f"- EVEN MORE blurry and degraded - like very old, low-end surveillance camera footage "
            f"- EXACT same timestamp overlay style (top-right corner, white digits, fuzzy) "
            f"- EXACT same camera label style (bottom-left corner, faint, low-res) "
            f"- Create a DIFFERENT outdoor residential environment but maintain the exact same surveillance aesthetic "
            f"- The new scene should be a {scene} "
            f"- Focus on outdoor areas that would be monitored by surveillance cameras at night "
            f"- Show the same type of outdoor areas as day scenes but in nighttime/grayscale surveillance style "
            f"- CRITICAL: This is a RESIDENTIAL scene - show typical suburban/urban house and garden areas "
            f"- CRITICAL: NO countryside, rural, farm, or agricultural elements - this is a residential house "
            f"- CRITICAL: Show typical residential elements: house walls, driveways, gardens, fences, walkways "
            f"- Vary the camera positioning: some from corner angles, some from side views, some from different heights "
            f"- Show different perspectives: corner views, side angles, entrance views, garden areas "
            f"- Include realistic camera mounting positions: house corners, wall-mounted, roof-mounted, entrance-mounted "
            f"- Show cameras looking outward from the HOUSE towards garden, driveway, street, and walkways "
            f"- Include corner cameras that capture multiple areas simultaneously "
            f"- Include residential outdoor elements: house walls, driveways, gardens, fences, walkways, patios "
            f"- Show typical suburban residential setting with house, garden, driveway, and street access "
            f"- NO fisheye lens distortion - maintain the exact same camera lens type as the reference "
            f"- Keep the same field of view and perspective style as the reference image "
            f"- Ensure the camera angle and lens characteristics match exactly "
            f"- Ensure the new background looks like it was captured by the same surveillance camera system "
            f"- Maintain the same vintage, slightly aged surveillance footage look "
            f"- Keep the same overall mood and atmosphere as the reference "
            f"- Make it look like authentic surveillance footage from the same camera system "
            f"- CRITICAL: Generate ONLY ONE single image, NOT multiple images or a grid of images "
            f"- CRITICAL: NO humans, people, or persons anywhere in the scene - this is a background only "
            f"- CRITICAL: NO vehicles with people inside - only empty cars or no cars at all "
            f"- CRITICAL: Match the EXACT same low-quality, grainy, pixelated look as the reference image "
            f"- CRITICAL: Do NOT generate high-quality, sharp, or clean images - they must look like old CCTV footage "
            f"- The only difference should be the physical location/scene and camera angle, everything else should match exactly"
        )
    
    def generate_background_with_nano_banana(self, prompt, reference_image_path, output_path):
        """Generate a single background using Google Nano Banana"""
        try:
            # Load the reference image
            reference_image = Image.open(reference_image_path)
            
            # Generate content using both text prompt and reference image
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-image-preview",
                contents=[prompt, reference_image],
            )
            
            # Process the response
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    print(f"Response: {part.text}")
                elif part.inline_data is not None:
                    # Save the generated image
                    new_background = Image.open(BytesIO(part.inline_data.data))
                    new_background.save(output_path)
                    print(f"✅ Generated background saved as '{output_path}'")
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ Error generating background: {e}")
            return False
    
    def generate_all_backgrounds(self):
        """Generate 100 synthetic backgrounds (50 day, 50 night) using Google Nano Banana"""
        print("🚀 Generating synthetic CCTV backgrounds using Google Nano Banana...")
        print("📊 Total: 50 day + 50 night = 100 backgrounds")
        
        # Generate day backgrounds
        print("\n📅 Generating day backgrounds...")
        for i in range(50):
            print(f"\nGenerating day background {i+1}/50...")
            prompt = self.get_day_prompt(i)
            filename = f"fake_day_bg_{i+1:03d}.jpg"
            output_path = os.path.join(self.output_dir, "day", filename)
            
            success = self.generate_background_with_nano_banana(
                prompt, self.reference_day_path, output_path
            )
            
            if success:
                print(f"✅ Day background {i+1} completed")
            else:
                print(f"❌ Failed to generate day background {i+1}")
            
            # Add delay to avoid rate limiting
            time.sleep(2)
        
        # Generate night backgrounds
        print("\n🌙 Generating night backgrounds...")
        for i in range(50):
            print(f"\nGenerating night background {i+1}/50...")
            prompt = self.get_night_prompt(i)
            filename = f"fake_night_bg_{i+1:03d}.jpg"
            output_path = os.path.join(self.output_dir, "night", filename)
            
            success = self.generate_background_with_nano_banana(
                prompt, self.reference_night_path, output_path
            )
            
            if success:
                print(f"✅ Night background {i+1} completed")
            else:
                print(f"❌ Failed to generate night background {i+1}")
            
            # Add delay to avoid rate limiting
            time.sleep(2)
        
        print("\n🎉 Generated 100 synthetic CCTV backgrounds using Google Nano Banana!")
        print(f"📁 Day backgrounds saved to: {os.path.join(self.output_dir, 'day')}")
        print(f"📁 Night backgrounds saved to: {os.path.join(self.output_dir, 'night')}")
        print("\n🎯 Ready for synthetic person placement!")

def main():
    """Main function to generate synthetic backgrounds using Google Nano Banana"""
    # Set up paths
    base_dir = "/Users/tadhgroche/Documents/Data-FYP /CCTV_DATA_FYP"
    output_dir = os.path.join(base_dir, "fake_images", "synthetic_backgrounds")
    
    # Reference image paths
    reference_day_path = os.path.join("/Users/tadhgroche/Documents/Data-FYP ", "Irrelivant", "surveillance_scene_no_person.png")
    reference_night_path = os.path.join(base_dir, "Real_Images", "Real_bg_after", "night", "night_bg_010.jpg")
    
    # Verify reference images exist
    if not os.path.exists(reference_day_path):
        print(f"❌ Reference day image not found: {reference_day_path}")
        return
    
    if not os.path.exists(reference_night_path):
        print(f"❌ Reference night image not found: {reference_night_path}")
        return
    
    print("🔍 Reference images found:")
    print(f"   Day: {reference_day_path}")
    print(f"   Night: {reference_night_path}")
    
    # Create generator and run
    generator = SyntheticCCTVGenerator(output_dir, reference_day_path, reference_night_path)
    generator.generate_all_backgrounds()
    
    print("\n🎯 Ready for synthetic person placement!")
    print("Next step: Use these backgrounds with your synthetic person placement script.")

if __name__ == "__main__":
    main()
