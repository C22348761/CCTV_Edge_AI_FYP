#!/usr/bin/env python3
"""
Fake Background Synthetic Person Placement Script
Adds synthetic people to fake CCTV backgrounds with proper naming and quality matching.
Each fake background gets multiple synthetic person variations with background ID tracking.
"""

from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
import os
from pathlib import Path
from dotenv import load_dotenv
import time
import random
import secrets
import re

# Load environment variables from .env file
load_dotenv()

# Create client (API key should be set as environment variable)
client = genai.Client()

# Base prompt for adding realistic synthetic people to fake backgrounds
base_prompt = (
    "Add exactly ONE realistic synthetic person to this surveillance camera scene. The person must: "
    "- Look completely natural and fit seamlessly into the existing environment "
    "- Have realistic body proportions and natural poses "
    "- Match the surveillance camera quality (slightly blurry, not high-resolution) "
    "- Have appropriate lighting and shadows that match the scene perfectly "
    "- Blend seamlessly with the existing lighting conditions and camera angle "
    "- Show natural surveillance camera characteristics (slight grain, desaturated colors) "
    "- Appear as if they were originally captured by the same surveillance camera "
    "- Maintain the same camera perspective and elevated angle as the background "
    "- Have realistic but not hyper-detailed facial features (surveillance quality) "
    "- Fit naturally into the existing scene without looking added or artificial "
    "- CRITICAL: Create either ONE person OR TWO people maximum - no more than 2 people total "
    "- CRITICAL: If creating 2 people, they should be walking together or interacting naturally "
    "- CRITICAL: The person should NOT be looking directly at the camera "
    "- The person should be looking away from the camera, at the ground, or in a natural direction "
    "- The person should appear unaware they are being filmed (natural surveillance behavior) "
    "- CRITICAL: Position the person near the EDGES or SIDES of the frame, NOT in the center "
    "- CRITICAL: Most surveillance captures people walking along edges, sides, or perimeters "
    "- CRITICAL: Avoid center-frame positioning - people are usually caught on the sides "
    "- CRITICAL: Person's size must be PROPORTIONAL to their distance from camera "
    "- CRITICAL: FOREGROUND person = LARGE (40-60% of frame height) - must look like a REAL PERSON with realistic features, not synthetic/fake "
    "- CRITICAL: UP-CLOSE people must have REALISTIC facial features BUT heavily degraded CCTV quality "
    "- CRITICAL: UP-CLOSE faces and upper bodies must be MORE PIXELATED, MORE GRAINY, MORE BLURRY than background "
    "- CRITICAL: UP-CLOSE people need HEAVY compression artifacts, digital noise, and blur on face/skin "
    "- CRITICAL: When you can see face clearly, make it EXTRA GRAINY and PIXELATED like low-quality CCTV "
    "- CRITICAL: Face and upper body should be HEAVILY degraded - not high quality or sharp "
    "- CRITICAL: MID-GROUND person = MEDIUM (20-30% of frame height) with moderate detail "
    "- CRITICAL: BACKGROUND person = SMALL (15-25% of frame height) with clear visibility but less detail "
    "- CRITICAL: Use realistic perspective scaling - people shrink with distance "
    "- CRITICAL: Match the perspective and scale of any existing objects in the background "
    "- CRITICAL: Person may be PARTIALLY CUT OFF by frame edges "
    "- CRITICAL: Position should suggest they're passing through camera's field of view "
    "- CRITICAL: The person must look like a REAL HUMAN being, not a fake/synthetic/artificial person "
    "- CRITICAL: Realistic skin texture, natural facial features, authentic clothing wrinkles and folds "
    "- CRITICAL: The person should look like an actual human caught on surveillance, not a generated image "
    "- CRITICAL: Natural human proportions, realistic body language, authentic appearance "
    "- CRITICAL: NO ARTIFACTS - person must have complete body parts (no missing feet, hands, legs, or body parts) "
    "- CRITICAL: If feet are visible, they must be complete and properly attached "
    "- CRITICAL: If hands are visible, they must be complete and properly formed "
    "- CRITICAL: No distorted or incomplete body parts - ensure anatomical correctness "
    "- CRITICAL: Person must be a complete, whole human figure without artifacts or glitches "
    "- CRITICAL: The person must have the EXACT SAME blur, grain, and compression as the background "
    "- CRITICAL: NOT sharper/cleaner than background - match the background's pixelation and noise level precisely "
    "- CRITICAL: NOT more degraded than background - match the background's quality level exactly "
    "- CRITICAL: The person should be indistinguishable in quality from the original background elements "
    "- CRITICAL: Add slight motion blur consistent with surveillance frame rates "
    "- CRITICAL: The person must blend into the CCTV footage style - grainy, slightly blurry, compressed "
    "- CRITICAL: Match the EXACT surveillance camera aesthetic - not too clean, not too degraded "
    "- CRITICAL: Make the person the SAME pixelation and degradation level as the background "
    "- CRITICAL: The person should look like they were captured by the SAME camera as the background "
    "- CRITICAL: Add appropriate motion blur, compression artifacts, and digital noise to match the background "
    "- CRITICAL: The person should have consistent surveillance quality - not too high, not too low "
    "- CRITICAL: For night scenes: make the person the SAME grainy and degraded level as the background "
    "- CRITICAL: Person must have shadows matching the EXACT light sources in scene "
    "- CRITICAL: NO studio lighting - match realistic poor surveillance lighting "
    "- CRITICAL: Skin and clothing should have same lighting problems as environment "
    "- CRITICAL: The person should look like they were added by a low-quality surveillance system "
    "- QUALITY CONSISTENCY CHECK: Before finalizing the person, verify they match the background's blur, grain, compression, and overall degradation "
    "- QUALITY CONSISTENCY CHECK: The person should NOT look like a sharp photograph pasted onto surveillance footage "
    "- QUALITY CONSISTENCY CHECK: Apply the EXACT SAME camera quality degradation to the person as exists in the background "
    "- QUALITY CONSISTENCY CHECK: If the deck railing is blurry, the person must be equally blurry "
    "- QUALITY CONSISTENCY CHECK: If the grass has grain, the person must have identical grain "
    "- QUALITY CONSISTENCY CHECK: The person should be INDISTINGUISHABLE in quality level from the background elements "
    "- QUALITY CONSISTENCY CHECK: Match the background's pixelation, noise, compression artifacts, and blur level exactly "
    "- QUALITY CONSISTENCY CHECK: The person must have the same surveillance camera quality as every other element in the scene "
)

# Night-specific quality adjustments (CRITICAL for grayscale/monochrome)
night_quality_prompt = (
    "CRITICAL: This is a NIGHT surveillance scene. The person must have: "
    "- COMPLETELY GRAYSCALE/MONOCHROME appearance - NO colors visible whatsoever "
    "- Only black, white, and gray tones - exactly like the background "
    "- Match the night CCTV style with appropriate grain and noise "
    "- The person should have the SAME level of grain and compression as the background "
    "- Darker, more muted tones that match night lighting conditions "
    "- Match the blur level and detail level of the background exactly "
    "- Consistent grain and noise that matches the background's quality "
    "- Match the contrast level of the night scene "
    "- Lighting on clothing and skin should match the background's lighting quality "
    "- Match the resolution and compression level of the surveillance background "
    "- The person should have the same focus and motion blur as background elements "
    "- CRITICAL: The person must look like a REALISTIC person with natural features "
    "- CRITICAL: Match the background's quality level - not better, not worse "
    "- CRITICAL: Apply the SAME blur, grain, and compression as the background shows "
    "- CRITICAL: The person should blend seamlessly with the night CCTV aesthetic "
    "- CRITICAL: Match the EXACT surveillance camera quality level of the background "
    "- CRITICAL: The person should have consistent night CCTV quality matching the scene "
    "- EXTRA CRITICAL FOR NIGHT: The person must NOT be looking at the camera "
    "- The person should be looking down, away, or in a natural direction "
    "- Night surveillance behavior: person appears completely unaware of being filmed "
    "- The person should be focused on their activity, not the camera "
    "- IMPORTANT: The person must match the background's quality - realistic but with CCTV characteristics "
    "- The person should blend seamlessly with the night surveillance aesthetic "
    "- CRITICAL: NO colors should be visible - only grayscale tones like the background "
    "- QUALITY CONSISTENCY CHECK: The person must match the background's blur, grain, and compression exactly "
    "- QUALITY CONSISTENCY CHECK: The person should NOT look like a sharp photograph pasted onto surveillance footage "
    "- QUALITY CONSISTENCY CHECK: Apply the EXACT SAME camera quality as exists in the background "
    "- QUALITY CONSISTENCY CHECK: If the background is blurry, the person must be equally blurry "
    "- QUALITY CONSISTENCY CHECK: If the background has grain, the person must have identical grain "
    "- QUALITY CONSISTENCY CHECK: The person should be INDISTINGUISHABLE in quality level from the background "
    "- QUALITY CONSISTENCY CHECK: Match the background's quality exactly - not too clean, not too degraded "
    "- QUALITY CONSISTENCY CHECK: The person must have the same surveillance camera quality as every other element in the scene "
)

def create_output_folders(base_path):
    """Create the fake synthetic person folder structure."""
    synthetic_path = base_path / "fake_images" / "fake_synthetic_person"
    day_path = synthetic_path / "day"
    night_path = synthetic_path / "night"
    
    # Create directories if they don't exist
    day_path.mkdir(parents=True, exist_ok=True)
    night_path.mkdir(parents=True, exist_ok=True)
    
    return day_path, night_path

def extract_background_id(filename):
    """Extract background ID from filename like 'fake_day_bg_001.jpg' -> '001'"""
    match = re.search(r'fake_(?:day|night)_bg_(\d+)', filename)
    if match:
        return match.group(1)
    return None

def get_diverse_prompt(is_night=False, background_id=None):
    """Generate a truly diverse prompt for adding synthetic people."""
    # Use cryptographically secure random for true randomness
    random.seed(secrets.randbelow(2**32))
    
    # Expanded person types with more variety
    person_types = [
        # Male variations
        "young man in casual clothing walking away from camera",
        "man in jeans and t-shirt walking towards camera",
        "young man in hoodie and jeans bending down",
        "man in casual jacket looking down at something",
        "middle-aged man in business casual walking",
        "man in shorts and t-shirt walking briskly",
        "man in jeans and flannel shirt standing still",
        "man in track pants and hoodie walking away",
        "man in hoodie with hood up walking stealthily",
        "man in dark clothing with hood up looking around",
        "man in hoodie with hood up walking near existing vehicle",
        "man in dark hoodie with hood up behind existing structure",
        "man in hoodie with hood up in the distance",
        "man in dark clothing with hood up walking along edge",
        "man in hoodie with hood up looking into existing vehicle",
        "man in dark clothing with hood up walking around existing vehicle",
        "man in hoodie with hood up standing near a vehicle",
        "man in dark clothing with hood up walking behind existing structure",
        "man in hoodie with hood up looking into a garden",
        "man in dark clothing with hood up walking along existing structure",
        "man in hoodie with hood up standing behind existing structure",
        "man in dark clothing with hood up walking in the distance",
        "man in hoodie with hood up appearing from around a corner",
        "man in dark clothing with hood up walking along the perimeter",
        "man in hoodie with hood up standing near a building",
        "man in dark clothing with hood up walking near existing structure",
        
        # Female variations  
        "young woman in casual clothing walking away from camera",
        "woman in jeans and sweater walking towards camera",
        "young woman in hoodie and leggings bending down",
        "woman in casual jacket looking down at phone",
        "middle-aged woman in business casual walking",
        "woman in dress and cardigan walking slowly",
        "woman in summer dress walking towards camera",
        "woman in yoga pants and tank top jogging",
        "woman in business suit walking purposefully",
        "woman in hoodie with hood up walking stealthily",
        "woman in dark clothing with hood up looking around",
        "woman in hoodie with hood up walking near existing vehicle",
        "woman in dark clothing with hood up behind existing structure",
        "woman in hoodie with hood up in the distance",
        "woman in dark clothing with hood up walking along edge",
        "woman in hoodie with hood up looking into existing vehicle",
        "woman in dark clothing with hood up walking around existing vehicle",
        "woman in hoodie with hood up standing near a vehicle",
        "woman in dark clothing with hood up walking behind existing structure",
        "woman in hoodie with hood up looking into a garden",
        "woman in dark clothing with hood up walking along existing structure",
        "woman in hoodie with hood up standing behind existing structure",
        "woman in dark clothing with hood up walking in the distance",
        "woman in hoodie with hood up appearing from around a corner",
        "woman in dark clothing with hood up walking along the perimeter",
        "woman in hoodie with hood up standing near a building",
        "woman in dark clothing with hood up walking near existing structure",
        
        # Group activities (2 people)
        "two people walking together - one man and one woman",
        "two friends walking side by side",
        "two people walking with hoodies up",
        "two people in dark clothing walking together",
        "two people walking near existing vehicle",
        "two people walking behind existing structure",
        "two people walking in the distance",
        "two people walking along the edge of the frame",
        "two people walking around a vehicle",
        "two people walking along existing structure",
        "two people walking near a building",
        "two people walking along the perimeter",
        "two people appearing from around a corner",
        "two people walking near existing structure",
        "two people walking along the side of the scene",
        "two people walking near the border",
        "two people walking in the background",
        "two people walking in the foreground near edge",
        "two people walking diagonally across the scene",
        
        # Generic person descriptions for maximum variety
        "person in casual clothing walking naturally",
        "person in everyday attire moving through the scene",
        "person in dark clothing walking along the edge",
        "person looking at their phone while walking",
        "two teenagers walking together",
        "person riding a bicycle through the scene",
        "person walking with headphones on",
        "person sitting on a bench or ledge",
        "person looking into existing vehicle window",
        "person walking around existing vehicle",
        "person standing behind existing structure looking into garden",
        "person walking along existing structure",
        "person in the distance walking slowly",
        "person appearing from behind a building",
        "person walking along existing structure",
        "person standing near a vehicle",
        "person walking near a gate or entrance",
        "person walking along a pathway",
        "person walking near existing structure",
        "person walking near a garden area",
        "person walking near a driveway",
        "person walking near a walkway",
        "person walking near a patio area",
        "person walking near a deck or porch",
        "person walking near a garage",
        "person walking near existing structure",
        "person walking near existing vegetation",
        "person walking near a mailbox",
        "person walking near a street sign",
        "person walking near a lamppost",
        "person walking near a trash can",
        "person walking near a recycling bin",
        "person walking near a flower bed",
        "person walking near existing vegetation",
        "person walking near existing structure",
        "person walking near existing structure",
        "person walking near a corner of a building",
        "person walking near a side entrance",
        "person walking near a back entrance",
        "person walking near a front entrance",
        "person walking near a side door",
        "person walking near a back door",
        "person walking near a front door",
        "person walking near a window",
        "person walking near a side window",
        "person walking near a back window",
        "person walking near a front window"
    ]
    
    selected_person = secrets.choice(person_types)
    
    # Clothing colors
    if is_night:
        clothing_colors = ["dark", "black", "gray", "very dark", "charcoal", "dark gray"]
    else:
        clothing_colors = ["dark", "black", "gray", "blue", "green", "red", "white", "navy", "brown", "tan", "beige"]
    
    clothing_items = ["hoodie", "jacket", "sweater", "t-shirt", "shirt", "dress", "pants", "jeans", "shorts", "coat"]
    
    random_color = secrets.choice(clothing_colors)
    random_item = secrets.choice(clothing_items)
    
    # Random activity locations - MASSIVELY EXPANDED for maximum variety
    # 10% FOREGROUND, 30% MID-GROUND, 60% BACKGROUND
    location_options = [
        # FOREGROUND (10% - rare, edge positioning only)
        "in the FOREGROUND partially cut off by left edge - person should be LARGE (40-60% of frame height)",
        "in the FOREGROUND partially cut off by right edge - person should be LARGE (40-60% of frame height)",
        "in the FOREGROUND at the bottom left corner partially visible - person should be LARGE (40-60% of frame height)",
        "in the FOREGROUND at the bottom right corner partially visible - person should be LARGE (40-60% of frame height)",
        "in the FOREGROUND walking along the very left edge - person should be LARGE (40-60% of frame height)",
        "in the FOREGROUND walking along the very right edge - person should be LARGE (40-60% of frame height)",
        
        # MID-GROUND (30% - medium person, varied edge positions)
        "in the MID-GROUND walking along the left edge of frame - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND walking along the right edge of frame - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND partially cut off by left edge - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND partially cut off by right edge - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND moving through left third of frame - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND moving through right third of frame - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND at the left perimeter - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND at the right perimeter - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND emerging from left side - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND emerging from right side - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND walking past on left side - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND walking past on right side - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND at left corner area - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND at right corner area - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND along left boundary - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND along right boundary - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND passing through left zone - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND passing through right zone - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND at left margin - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND at right margin - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND moving along left perimeter - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND moving along right perimeter - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND half cut off by left frame edge - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND half cut off by right frame edge - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND walking near left border - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND walking near right border - person should be MEDIUM size (20-30% of frame height)",
        
        # MID-GROUND PARTIALLY VISIBLE BEHIND EXISTING OBJECTS (Priority scenarios)
        "in the MID-GROUND with only HEAD and SHOULDERS visible behind existing object - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND with only UPPER BODY visible behind existing structure - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND with 50% of body hidden behind whatever exists in scene - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND with 60% of body hidden behind existing elements - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND with 70% of body hidden behind existing objects - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND partially obscured by existing structures with only 40% visible - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND partially obscured by existing elements with only 30% visible - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND walking behind existing objects with legs hidden - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND walking behind existing structures with lower body hidden - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND peeking around existing object with most of body hidden - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND standing behind existing element with only head/torso visible - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND crouching behind existing structure with only upper body visible - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND leaning around existing object partially hidden - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND emerging from behind existing structure half visible - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND ducking behind existing element partially obscured - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND positioned behind existing objects with partial visibility - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND hiding behind existing structure with only portion visible - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND moving behind existing elements partially hidden - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND walking behind existing objects on the left partially visible - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND walking behind existing objects on the right partially visible - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND behind existing structure on left with 60% hidden - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND behind existing structure on right with 60% hidden - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND behind existing elements with only top half visible - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND behind existing objects with only side visible - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND partially blocked by existing structure - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND partially covered by existing elements - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND standing behind existing object showing only 40% of body - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND standing behind existing structure showing only 50% of body - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND walking past existing object with partial occlusion - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND moving behind existing barrier partially hidden - person should be MEDIUM size (20-30% of frame height)",
        
        # BACKGROUND (60% - small person, far from camera, maximum variety)
        "in the BACKGROUND walking in the far distance - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking along a distant path - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking between distant structures - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking along a distant boundary - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND appearing as a small figure in the distance - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking in the far background - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking along the far left edge - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking along the far right edge - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND partially cut off by far left edge - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND partially cut off by far right edge - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND emerging from far left edge - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND emerging from far right edge - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking through far left area - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking through far right area - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND at the far left corner - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND at the far right corner - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND moving along distant left perimeter - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND moving along distant right perimeter - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking in far left zone - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking in far right zone - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND as a small figure on left side - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND as a small figure on right side - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking along distant left boundary - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking along distant right boundary - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND in the far left distance - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND in the far right distance - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND moving through far left region - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND moving through far right region - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking near far left edge - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking near far right edge - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND at distant left margin - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND at distant right margin - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking through distant left side - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking through distant right side - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND as tiny figure on far left - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND as tiny figure on far right - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND moving along far left edge - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND moving along far right edge - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking in distant left area - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking in distant right area - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND at far left perimeter - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND at far right perimeter - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking behind whatever objects already exist in the scene - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND partially hidden behind existing structures - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking in the neighbor's garden in the distance - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND in the neighbor's yard on the left - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND in the neighbor's yard on the right - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND at the very edge of the house in the distance - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND behind the house in the far distance - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND around the side of the house - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND near the back corner of the house - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND at the far side of the property - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking along the neighbor's driveway - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND in the neighbor's backyard area - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND near the property line in the distance - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking between houses in the distance - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND behind existing vegetation - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND partially obscured by existing elements - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking near the far fence line - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND at the distant back of the property - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND near the side gate in the distance - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking along the distant edge of the yard - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND at the corner of the property line - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking through the distant yard area - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND near the back entrance in the distance - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking along the far side boundary - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND in the distant side yard area - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND near the far corner of the building - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking through the neighbor's property - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND at the distant edge of the frame - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND behind existing outdoor elements - person should be SMALL (15-25% of frame height)",
        
        # BACKGROUND PARTIALLY VISIBLE BEHIND EXISTING OBJECTS
        "in the BACKGROUND with only head visible behind existing objects in distance - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND with only upper body visible behind distant structures - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND with 70% hidden behind distant existing elements - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND partially obscured by distant existing objects - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking behind distant existing structures with partial visibility - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND peeking from behind distant existing elements - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND emerging from behind distant existing objects - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND standing behind distant existing structure with only portion visible - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND crouching behind distant existing elements partially hidden - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND moving behind distant existing objects with partial occlusion - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND walking past distant existing structures partially obscured - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND behind distant existing objects on left partially visible - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND behind distant existing objects on right partially visible - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND partially blocked by distant existing elements - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND partially covered by distant existing structures - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND with only silhouette visible behind distant objects - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND with only outline visible behind distant structures - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND ducking behind distant existing elements - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND leaning around distant existing objects - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND hiding behind distant existing structures partially visible - person should be SMALL (15-25% of frame height)",
        
        # MORE BACKGROUND VARIATIONS - Left side specific
        "in the BACKGROUND far left walking towards back - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND far left walking away - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND far left standing still - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND far left corner moving slowly - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND far left edge walking parallel - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND far left side walking diagonally - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND far left area crouching - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND far left zone bending over - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND far left margin standing - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND far left boundary walking slowly - person should be SMALL (15-25% of frame height)",
        
        # MORE BACKGROUND VARIATIONS - Right side specific
        "in the BACKGROUND far right walking towards back - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND far right walking away - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND far right standing still - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND far right corner moving slowly - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND far right edge walking parallel - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND far right side walking diagonally - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND far right area crouching - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND far right zone bending over - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND far right margin standing - person should be SMALL (15-25% of frame height)",
        "in the BACKGROUND far right boundary walking slowly - person should be SMALL (15-25% of frame height)",
        
        # MORE MID-GROUND LEFT variations
        "in the MID-GROUND left edge walking fast - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND left edge walking slow - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND left edge standing - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND left edge crouching - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND left edge bending - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND left side walking towards camera - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND left side walking away from camera - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND left corner area standing still - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND left corner area moving - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND left third walking diagonally - person should be MEDIUM size (20-30% of frame height)",
        
        # MORE MID-GROUND RIGHT variations  
        "in the MID-GROUND right edge walking fast - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND right edge walking slow - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND right edge standing - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND right edge crouching - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND right edge bending - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND right side walking towards camera - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND right side walking away from camera - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND right corner area standing still - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND right corner area moving - person should be MEDIUM size (20-30% of frame height)",
        "in the MID-GROUND right third walking diagonally - person should be MEDIUM size (20-30% of frame height)",
        
        # EXTREME DISTANCE (more variety)
        "in the EXTREME BACKGROUND as a tiny figure in the far distance - person should be VERY SMALL (10-15% of frame height)",
        "in the EXTREME BACKGROUND walking in the very far distance - person should be VERY SMALL (10-15% of frame height)",
        "in the EXTREME BACKGROUND appearing as a small dot in the distance - person should be VERY SMALL (10-15% of frame height)",
        "in the EXTREME BACKGROUND on the far left as a tiny dot - person should be VERY SMALL (10-15% of frame height)",
        "in the EXTREME BACKGROUND on the far right as a tiny dot - person should be VERY SMALL (10-15% of frame height)",
        "in the EXTREME BACKGROUND walking along the horizon on left - person should be VERY SMALL (10-15% of frame height)",
        "in the EXTREME BACKGROUND walking along the horizon on right - person should be VERY SMALL (10-15% of frame height)",
        "in the EXTREME BACKGROUND barely visible on far left - person should be VERY SMALL (10-15% of frame height)",
        "in the EXTREME BACKGROUND barely visible on far right - person should be VERY SMALL (10-15% of frame height)",
        "in the EXTREME BACKGROUND as a speck in the distance on left - person should be VERY SMALL (10-15% of frame height)",
        "in the EXTREME BACKGROUND as a speck in the distance on right - person should be VERY SMALL (10-15% of frame height)"
    ]
    
    # Activity options with 70% NORMAL / 30% SUSPICIOUS split
    # NORMAL ACTIVITIES (70% of generations) - MASSIVELY EXPANDED for maximum variety
    normal_activities = [
        # Walking variations - different speeds, directions, postures
        "walking naturally and casually while looking down or away",
        "walking quickly with purpose while looking at the ground", 
        "walking slowly while looking around at the environment",
        "walking with their hands in pockets, head down",
        "walking with arms swinging naturally, looking ahead",
        "walking with their head tilted down in thought",
        "walking while looking at the ground",
        "walking while looking at their feet",
        "walking briskly with confident stride",
        "walking leisurely with relaxed posture",
        "walking hesitantly while looking around",
        "walking determinedly while looking straight ahead",
        "strolling casually with hands at sides",
        "striding purposefully while looking down",
        "ambling slowly while glancing around",
        "marching quickly while looking forward",
        "wandering aimlessly while looking at surroundings",
        "pacing steadily while head is down",
        "stepping carefully while watching ground",
        "moving briskly while arms swing",
        
        # Different poses and body positions
        "standing still while looking at phone",
        "standing with arms crossed looking away",
        "standing with one hand on hip looking down",
        "standing casually with weight on one leg",
        "leaning against something while looking at phone",
        "bending down to pick something up",
        "crouching down while looking at ground",
        "kneeling while doing something",
        "sitting on steps while looking at phone",
        "sitting casually while waiting",
        "stretching arms while standing",
        "adjusting clothing while walking",
        "tying shoe while bent over",
        "checking watch while standing",
        "looking up at sky briefly",
        "shielding eyes from sun",
        "wiping face with hand",
        "scratching head while thinking",
        
        # Carrying/holding different items
        "walking with shopping bags while looking down",
        "carrying packages while looking at ground",
        "carrying groceries while looking at ground",
        "carrying laundry while looking at ground",
        "carrying boxes while looking at ground",
        "carrying tools while looking at ground",
        "carrying backpack while walking",
        "carrying briefcase while walking",
        "carrying purse while walking",
        "holding umbrella while walking",
        "carrying sports equipment while walking",
        "holding coffee cup while walking",
        "carrying bag over shoulder while walking",
        
        # Phone/device usage
        "talking on phone while walking and looking away",
        "texting on phone while walking slowly",
        "looking at phone while standing still",
        "scrolling on phone while walking",
        "holding phone to ear while standing",
        "checking phone while paused",
        
        # Normal residential activities
        "getting mail while looking at mailbox",
        "walking dog on leash while looking at ground",
        "taking out trash while looking at trash can",
        "arriving home while looking at keys",
        "doing yard work while looking at plants",
        "chatting with neighbor while looking at them",
        "unlocking car while looking at car door",
        "checking plants while looking at garden",
        "walking to mailbox while looking ahead",
        "walking to front door while looking at keys",
        "checking mail while looking at letters",
        "walking to car while looking at ground",
        "putting away trash cans while looking at ground",
        "opening gate while looking at latch",
        "closing gate while looking at handle",
        "watering plants while looking at garden",
        "sweeping pathway while looking at ground",
        "raking leaves while looking at ground",
        "trimming plants while bent over",
        "picking up items from ground",
        
        # Different directional movements
        "walking diagonally across the scene",
        "walking parallel to the camera view",
        "walking perpendicular to camera",
        "walking towards the camera angle",
        "walking away from the camera angle",
        "walking from left to right across frame",
        "walking from right to left across frame",
        "circling around an area",
        "pacing back and forth",
        "turning around while walking",
        "changing direction mid-walk",
        "pausing mid-stride then continuing",
        
        # Interacting with environment
        "opening door while looking at handle",
        "closing door while looking at door",
        "stepping over something carefully",
        "walking up steps while watching feet",
        "walking down steps while watching feet",
        "stepping onto pathway",
        "stepping off curb carefully",
        "navigating around obstacles",
        "sidestepping around something",
        "backing up slowly"
    ]
    
    # SUSPICIOUS ACTIVITIES (30% of generations)
    suspicious_activities = [
        # Suspicious behaviors
        "walking slowly while constantly looking around nervously",
        "stopping frequently to look behind them",
        "trying to stay in shadows or out of sight",
        "repeatedly approaching then retreating from door/window",
        "looking up at cameras or security features",
        "testing door handles or locks",
        "peering into windows from close distance",
        "loitering without clear purpose",
        "pacing in small area repeatedly",
        "crouching by doors or windows",
        "moving with exaggerated stealth",
        "obscuring face while moving",
        "examining security features",
        "carrying tools inappropriately (bolt cutters, crowbar)",
        "walking stealthily while looking around nervously",
        "moving slowly and cautiously while checking surroundings",
        "walking while constantly looking over their shoulder",
        "moving furtively while looking in all directions",
        "walking while trying to stay in shadows or out of view",
        "moving quietly while keeping their head down",
        "walking while appearing to hide their face",
        "moving suspiciously while looking around for cameras",
        "walking while appearing to case the area",
        "moving while looking like they don't belong there",
        "walking while appearing to be avoiding detection",
        "moving while looking like they're up to something",
        "walking while appearing nervous or jumpy",
        "moving while looking like they're trying to be unnoticed",
        "walking while appearing to scope out the area",
        "walking stealthily near existing vehicle while looking around",
        "moving cautiously behind existing structure while checking surroundings",
        "walking suspiciously in the distance while looking around",
        "moving furtively near a vehicle while looking over shoulder",
        "walking stealthily along existing structure while looking around",
        "moving cautiously near existing vehicle while looking around nervously",
        "walking suspiciously behind existing structure while looking around",
        "moving furtively in the distance while looking around",
        "walking stealthily near a building while looking around",
        "moving cautiously along existing structure while looking around",
        "walking suspiciously near existing structure while looking around",
        "moving furtively near a door while looking around",
        "walking stealthily near a window while looking around",
        "moving cautiously near a driveway while looking around",
        "walking suspiciously near a walkway while looking around"
    ]
    
    # Select activity type based on 70/30 split
    if secrets.randbelow(100) < 70:  # 70% normal
        activity_options = normal_activities
    else:  # 30% suspicious
        activity_options = suspicious_activities
    
    # Build the full prompt with maximum variety
    full_prompt = base_prompt + f"- {selected_person} "
    full_prompt += f"- wearing a {random_color} {random_item} "
    full_prompt += f"- positioned {secrets.choice(location_options)} "
    full_prompt += f"- {secrets.choice(activity_options)} "
    
    # Add random environmental context
    context_options = [
        "Choose clothing and activities that are contextually appropriate for the specific environment shown",
        "Ensure the person looks like they naturally belong in this specific surveillance scene",
        "Make sure the person's clothing and behavior match the time of day and setting",
        "The person should appear as if they were originally captured in this exact location"
    ]
    
    full_prompt += f"- {secrets.choice(context_options)}"
    
    # Add night-specific quality adjustments
    if is_night:
        full_prompt += f"\n\n{night_quality_prompt}"
    
    return full_prompt

def process_image(image_path, output_path, prefix, background_id, variation_id, is_night=False):
    """Process a single image by adding a synthetic person."""
    try:
        print(f"  Processing {image_path.name} (variation {variation_id})...")
        
        # Generate diverse prompt for this image
        prompt = get_diverse_prompt(is_night=is_night, background_id=background_id)
        
        # Load the input background image
        image = Image.open(image_path)
        
        # Generate content using both text prompt and background image
        response = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=[prompt, image],
        )
        
        # Process the response
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                # Save the generated image with synthetic person
                # Format: fake_day_synthetic_001_001.jpg (background_id_variation_id)
                output_filename = f"fake_{prefix}_synthetic_{background_id}_{variation_id:03d}.jpg"
                output_filepath = output_path / output_filename
                scene_with_person = Image.open(BytesIO(part.inline_data.data))
                scene_with_person.save(output_filepath)
                print(f"    ✅ Saved: {output_filename}")
                return True
        
        print(f"    ❌ No image data returned for {image_path.name}")
        return False
        
    except Exception as e:
        print(f"    ❌ Error processing {image_path.name}: {e}")
        return False

def get_next_image_id(output_path, background_id, prefix):
    """Get the next available variation ID for a background - enables resume functionality."""
    existing_files = list(output_path.glob(f"fake_{prefix}_synthetic_{background_id}_*.jpg"))
    if not existing_files:
        return 1
    
    # Extract variation IDs and find the next one
    variation_ids = []
    for file in existing_files:
        match = re.search(r'_(\d{3})\.jpg$', file.name)
        if match:
            variation_ids.append(int(match.group(1)))
    
    if variation_ids:
        return max(variation_ids) + 1
    return 1

def process_folder_with_variations(input_folder, output_folder, prefix, is_night=False, variations_per_bg=100):
    """Process all backgrounds in a folder, creating multiple variations per background."""
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Get all background images
    bg_images = sorted(input_path.glob("fake_*_bg_*.jpg"))
    
    if not bg_images:
        print(f"❌ No background images found in {input_folder}")
        return
    
    print(f"\n{'='*60}")
    print(f"Processing {len(bg_images)} backgrounds ({prefix})")
    print(f"Target: {variations_per_bg} variations per background")
    print(f"{'='*60}\n")
    
    total_generated = 0
    
    for bg_image in bg_images:
        background_id = extract_background_id(bg_image.name)
        if not background_id:
            print(f"⚠️  Skipping {bg_image.name} - couldn't extract background ID")
            continue
        
        print(f"\n📸 Background: {bg_image.name} (ID: {background_id})")
        
        # Get next variation ID (for resume functionality)
        next_variation_id = get_next_image_id(output_path, background_id, prefix)
        
        if next_variation_id > variations_per_bg:
            print(f"  ✅ Already complete ({next_variation_id-1}/{variations_per_bg} variations)")
            continue
        
        print(f"  Starting from variation {next_variation_id}/{variations_per_bg}")
        
        # Generate remaining variations
        for variation_id in range(next_variation_id, variations_per_bg + 1):
            success = process_image(
                image_path=bg_image,
                output_path=output_path,
                prefix=prefix,
                background_id=background_id,
                variation_id=variation_id,
                is_night=is_night
            )
            
            if success:
                total_generated += 1
            
            # Rate limiting to avoid API throttling
            time.sleep(2)
    
    print(f"\n{'='*60}")
    print(f"✅ Complete! Generated {total_generated} new images")
    print(f"{'='*60}\n")

def main():
    # Setup paths
    base_dir = Path("/Users/tadhgroche/Documents/Data-FYP ")
    fake_images_dir = base_dir / "CCTV_DATA_FYP" / "fake_images"
    
    # Create output folders
    day_output, night_output = create_output_folders(base_dir / "CCTV_DATA_FYP")
    
    # Input folders - backgrounds are in synthetic_backgrounds subfolder
    day_input = fake_images_dir / "synthetic_backgrounds" / "day"
    night_input = fake_images_dir / "synthetic_backgrounds" / "night"
    
    print("\n" + "="*60)
    print("FAKE BACKGROUND SYNTHETIC PERSON PLACEMENT")
    print("="*60)
    
    # Process day images (target: 65 variations per background = 3250 total)
    # Extra 5 variations to account for API error gaps
    if day_input.exists():
        process_folder_with_variations(
            input_folder=day_input,
            output_folder=day_output,
            prefix="day",
            is_night=False,
            variations_per_bg=80
        )
    
    # Process night images (target: 65 variations per background = 3250 total)
    # Extra 5 variations to account for API error gaps
    if night_input.exists():
        process_folder_with_variations(
            input_folder=night_input,
            output_folder=night_output,
            prefix="night",
            is_night=True,
            variations_per_bg=80
        )
    
    print("\n🎉 All processing complete!")

if __name__ == "__main__":
    main()

