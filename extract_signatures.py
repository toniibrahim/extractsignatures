#!/usr/bin/env python3
"""
PDF Signature Extractor
Extracts handwritten signatures from PDFs using OpenAI GPT-4o Vision API
"""

import os
import re
import json
import base64
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from pdf2image import convert_from_path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to load manual configuration overrides
try:
    from config import (
        MANUAL_BBOX_OVERRIDES,
        DEFAULT_BBOX,
        PRESERVE_COLORS,
        CLEANING_METHOD,
        OUTPUT_QUALITY,
        OUTPUT_DPI,
        OUTPUT_FORMAT,
        TRANSPARENT_BACKGROUND,
        MAKE_SQUARE,
        OUTPUT_SIZE,
        SIGNATURE_POSITION,
        SIGNATURE_PADDING,
        BOTTOM_MARGIN
    )
except ImportError:
    MANUAL_BBOX_OVERRIDES = {}
    DEFAULT_BBOX = None
    PRESERVE_COLORS = True
    CLEANING_METHOD = "color"
    OUTPUT_QUALITY = 100
    OUTPUT_DPI = 300
    OUTPUT_FORMAT = "png"
    TRANSPARENT_BACKGROUND = True
    MAKE_SQUARE = True
    OUTPUT_SIZE = 800
    SIGNATURE_POSITION = "bottom"
    SIGNATURE_PADDING = 0.05
    BOTTOM_MARGIN = 0.02

class SignatureExtractor:
    """Extract signatures from PDF documents using OpenAI Vision API"""

    def __init__(self, input_folder: str = "./input", output_folder: str = "./output", debug: bool = True):
        """
        Initialize the signature extractor

        Args:
            input_folder: Folder containing PDF files
            output_folder: Folder to save extracted signatures
            debug: Save debug images showing detection process
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.temp_folder = Path("./temp")
        self.debug_folder = Path("./debug")
        self.debug = debug

        # Create output and temp directories
        self.output_folder.mkdir(exist_ok=True)
        self.temp_folder.mkdir(exist_ok=True)
        if self.debug:
            self.debug_folder.mkdir(exist_ok=True)

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=api_key)

    def print_step(self, step_num: int, total_steps: int, message: str):
        """
        Print a formatted step message with progress indicator

        Args:
            step_num: Current step number
            total_steps: Total number of steps
            message: Message to display
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        progress = f"[{step_num}/{total_steps}]"
        print(f"  {timestamp} {progress} {message}")

    def print_progress_bar(self, current: int, total: int, prefix: str = "", length: int = 40):
        """
        Print a progress bar

        Args:
            current: Current item number
            total: Total number of items
            prefix: Prefix text
            length: Length of the progress bar
        """
        percent = 100 * (current / float(total))
        filled = int(length * current // total)
        bar = '█' * filled + '-' * (length - filled)
        print(f'\r{prefix} |{bar}| {percent:.1f}% ({current}/{total})', end='', flush=True)
        if current == total:
            print()  # New line when complete

    def extract_employee_number(self, filename: str) -> Optional[str]:
        """
        Extract employee number from filename

        Args:
            filename: PDF filename

        Returns:
            Employee number if found, None otherwise
        """
        # Try to find a number at the start of the filename
        match = re.match(r'^(\d+)', filename)
        if match:
            return match.group(1)

        # Try to find any number in the filename
        match = re.search(r'(\d+)', filename)
        if match:
            return match.group(1)

        return None

    def pdf_to_image(self, pdf_path: Path, step_num: int = 1, total_steps: int = 5) -> Path:
        """
        Convert the last page of PDF to high-quality JPEG

        Args:
            pdf_path: Path to PDF file
            step_num: Current step number
            total_steps: Total number of steps

        Returns:
            Path to generated JPEG file
        """
        self.print_step(step_num, total_steps, f"Converting PDF to image (300 DPI)...")
        start_time = time.time()

        try:
            # Auto-detect poppler path on Windows if needed
            poppler_path = None
            if os.name == 'nt':  # Windows
                # Common Windows poppler locations
                possible_paths = [
                    r"C:\Program Files\poppler\Library\bin",
                    r"C:\Program Files (x86)\poppler\Library\bin",
                    r"C:\poppler\Library\bin",
                    os.path.expanduser(r"~\poppler\Library\bin"),
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        poppler_path = path
                        print(f"      → Found poppler at: {poppler_path}")
                        break

            # Convert PDF to images (last page only)
            # Note: -1 for last page doesn't work, need to get page count first
            print(f"      → Reading PDF and converting last page...")

            kwargs = {
                'pdf_path': pdf_path,
                'dpi': 300,
                'fmt': 'jpeg',
                'last_page': None,  # Will convert all and take last
            }

            if poppler_path:
                kwargs['poppler_path'] = poppler_path

            images = convert_from_path(**kwargs)

            if not images or len(images) == 0:
                raise ValueError(f"No pages found in PDF: {pdf_path}")

            # Take the last page
            last_page = images[-1]
            print(f"      → Extracted page {len(images)} of {len(images)} (last page)")

            # Save the last page as JPEG
            temp_image_path = self.temp_folder / f"{pdf_path.stem}_temp.jpg"
            last_page.save(temp_image_path, 'JPEG', quality=95)

            elapsed = time.time() - start_time
            file_size = temp_image_path.stat().st_size / 1024  # KB
            print(f"      ✓ Converted to JPEG ({file_size:.1f} KB) in {elapsed:.2f}s")

            return temp_image_path

        except Exception as e:
            print(f"\n      ✗ PDF conversion failed!")
            print(f"      Error: {e}")

            if os.name == 'nt':  # Windows
                print(f"\n      💡 WINDOWS POPPLER INSTALLATION:")
                print(f"      1. Download from: https://github.com/oschwartz10612/poppler-windows/releases/")
                print(f"      2. Extract to: C:\\Program Files\\poppler")
                print(f"      3. Add to PATH: C:\\Program Files\\poppler\\Library\\bin")
                print(f"      4. Restart your terminal")
                print(f"\n      Or install via Chocolatey: choco install poppler")
            else:
                print(f"\n      💡 LINUX/MAC POPPLER INSTALLATION:")
                print(f"      Ubuntu/Debian: sudo apt-get install poppler-utils")
                print(f"      macOS: brew install poppler")

            raise

    def encode_image(self, image_path: Path) -> str:
        """
        Encode image to base64 for OpenAI API

        Args:
            image_path: Path to image file

        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_signature_bbox_from_openai(self, image_path: Path, employee_number: Optional[str] = None,
                                       step_num: int = 2, total_steps: int = 5) -> dict:
        """
        Use OpenAI GPT-4o to detect signature bounding box

        Args:
            image_path: Path to image file
            employee_number: Known employee number (optional)
            step_num: Current step number
            total_steps: Total number of steps

        Returns:
            Dictionary with bbox coordinates and employee number
        """
        self.print_step(step_num, total_steps, "Analyzing image with OpenAI GPT-4o (latest) Vision API...")
        start_time = time.time()

        # Check for manual bbox override
        manual_bbox = None
        if employee_number and employee_number in MANUAL_BBOX_OVERRIDES:
            manual_bbox = MANUAL_BBOX_OVERRIDES[employee_number]
            print(f"      ℹ Using manual bbox override for employee {employee_number}")
        elif DEFAULT_BBOX is not None:
            manual_bbox = DEFAULT_BBOX
            print(f"      ℹ Using default manual bbox (from config.py)")

        if manual_bbox:
            result = {
                'signature_bbox': manual_bbox,
                'employee_number': employee_number
            }
            elapsed = time.time() - start_time
            bbox = manual_bbox
            print(f"      ✓ Manual bounding box: ymin={bbox['ymin']:.3f}, xmin={bbox['xmin']:.3f}, ymax={bbox['ymax']:.3f}, xmax={bbox['xmax']:.3f}")
            print(f"      ✓ Skipped AI detection (using manual override) in {elapsed:.2f}s")

            # Save debug visualization if enabled
            if self.debug:
                self.save_bbox_visualization(image_path, bbox, employee_number)

            return result

        # Encode image
        print(f"      → Encoding image to base64...")
        base64_image = self.encode_image(image_path)
        encoded_size = len(base64_image) / 1024  # KB
        print(f"      → Sending image to OpenAI ({encoded_size:.1f} KB)...")

        # Create enhanced prompt with step-by-step reasoning
        prompt = """CRITICAL TASK: Locate the HANDWRITTEN SIGNATURE on this form.

VISUAL IDENTIFICATION STEPS:
1. Find the text "Handwritten Signature Specimen:" near the bottom of the page (this is PRINTED text)
2. The HANDWRITTEN SIGNATURE is the cursive/script writing that appears IN THE LARGE BOX BELOW this label
3. The signature consists of flowing, connected handwritten pen strokes (cursive writing)
4. It will look distinctly different from the typed/printed text on the form

IMPORTANT - WHAT TO LOOK FOR:
- HANDWRITTEN cursive/script marks (NOT printed text)
- Located INSIDE a box/field below the "Handwritten Signature Specimen:" label
- Appears in the BOTTOM THIRD of the page (but NOT at the very bottom edge)
- Typical Y-coordinates: between 0.75 and 0.92 (NOT 0.95-1.0)
- If you see a signature, it's usually around y=0.82 to y=0.90

COMMON MISTAKES TO AVOID:
❌ DO NOT select empty space below the signature
❌ DO NOT select the bottom edge of the page
❌ DO NOT select the label text "Handwritten Signature Specimen:"
❌ DO NOT select printed text
✅ DO select the actual handwritten cursive marks

VALIDATION:
- The signature bbox should have: ymin between 0.75-0.90 (NOT above 0.90)
- Width should be reasonable (xmax - xmin should be 0.3 to 0.7)
- Height should be compact (ymax - ymin should be 0.03 to 0.15)

Also extract the Employee Number from the top portion of the form.

Return ONLY this JSON (no other text):
{
    "signature_bbox": {
        "ymin": <0.75 to 0.90>,
        "xmin": <0.1 to 0.4>,
        "ymax": <0.82 to 0.95>,
        "xmax": <0.4 to 0.9>
    },
    "employee_number": "<number or null>"
}

EXAMPLE for a signature in the middle-bottom area:
{
    "signature_bbox": {
        "ymin": 0.82,
        "xmin": 0.15,
        "ymax": 0.89,
        "xmax": 0.68
    },
    "employee_number": "3905"
}"""

        try:
            # Use latest GPT-4o model without max_tokens parameter
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Using latest available model
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                temperature=0  # Removed max_tokens parameter
            )

            # Parse response
            response_text = response.choices[0].message.content.strip()
            print(f"      → Raw API response length: {len(response_text)} chars")

            # Remove markdown code blocks if present
            response_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()

            result = json.loads(response_text)

            # Use provided employee number if OpenAI couldn't find it
            if employee_number and not result.get('employee_number'):
                result['employee_number'] = employee_number

            elapsed = time.time() - start_time
            bbox = result['signature_bbox']
            print(f"      ✓ Received bounding box: ymin={bbox['ymin']:.3f}, xmin={bbox['xmin']:.3f}, ymax={bbox['ymax']:.3f}, xmax={bbox['xmax']:.3f}")

            # Validate bounding box
            warnings = []
            bbox_height = bbox['ymax'] - bbox['ymin']
            bbox_width = bbox['xmax'] - bbox['xmin']

            # Check if bbox seems incorrect
            if bbox['ymin'] > 0.92:
                warnings.append(f"ymin too low ({bbox['ymin']:.3f} > 0.92) - signature might be missed")
            if bbox['ymax'] > 0.97:
                warnings.append(f"ymax too close to bottom ({bbox['ymax']:.3f} > 0.97) - might be detecting empty space")
            if bbox_height < 0.02:
                warnings.append(f"height too small ({bbox_height:.3f} < 0.02) - signature might be cut off")
            if bbox_height > 0.25:
                warnings.append(f"height too large ({bbox_height:.3f} > 0.25) - might include extra content")
            if bbox_width < 0.15:
                warnings.append(f"width too small ({bbox_width:.3f} < 0.15) - signature might be cut off")

            if warnings:
                print(f"      ⚠ VALIDATION WARNINGS:")
                for warning in warnings:
                    print(f"        • {warning}")
                print(f"      ⚠ The detected bbox may be incorrect. Check debug images!")
            else:
                print(f"      ✓ Bounding box validation passed")

            if result.get('employee_number'):
                print(f"      ✓ Detected employee number: {result['employee_number']}")
            print(f"      ✓ API call completed in {elapsed:.2f}s")

            # Save debug visualization if enabled
            if self.debug:
                self.save_bbox_visualization(image_path, bbox, employee_number)

            return result

        except Exception as e:
            print(f"      ✗ Error calling OpenAI API: {e}")
            raise

    def save_bbox_visualization(self, image_path: Path, bbox: dict, emp_num: Optional[str] = None):
        """
        Save a debug image showing the detected bounding box

        Args:
            image_path: Path to the original image
            bbox: Bounding box dictionary
            emp_num: Employee number for filename
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return

            height, width = img.shape[:2]

            # Convert normalized coordinates to pixels
            ymin = int(bbox['ymin'] * height)
            xmin = int(bbox['xmin'] * width)
            ymax = int(bbox['ymax'] * height)
            xmax = int(bbox['xmax'] * width)

            # Draw rectangle
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)

            # Add text label
            label = f"Signature Detection"
            cv2.putText(img, label, (xmin, ymin - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Add coordinates text
            coords_text = f"({xmin}, {ymin}) to ({xmax}, {ymax})"
            cv2.putText(img, coords_text, (xmin, ymax + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Save debug image
            filename = f"debug_{emp_num or 'unknown'}_bbox.jpg"
            debug_path = self.debug_folder / filename
            cv2.imwrite(str(debug_path), img)
            print(f"      💾 Debug visualization saved: {debug_path.name}")

        except Exception as e:
            print(f"      ⚠ Could not save debug visualization: {e}")

    def crop_signature(self, image_path: Path, bbox: dict, step_num: int = 3, total_steps: int = 5) -> np.ndarray:
        """
        Crop signature area from image using bounding box

        Args:
            image_path: Path to image file
            bbox: Bounding box dictionary with normalized coordinates
            step_num: Current step number
            total_steps: Total number of steps

        Returns:
            Cropped image as numpy array
        """
        self.print_step(step_num, total_steps, "Cropping signature area...")
        start_time = time.time()

        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        height, width = img.shape[:2]
        print(f"      → Original image size: {width}x{height} pixels")

        # Convert normalized coordinates to pixels
        ymin = int(bbox['ymin'] * height)
        xmin = int(bbox['xmin'] * width)
        ymax = int(bbox['ymax'] * height)
        xmax = int(bbox['xmax'] * width)

        # Ensure coordinates are within image bounds
        ymin = max(0, ymin)
        xmin = max(0, xmin)
        ymax = min(height, ymax)
        xmax = min(width, xmax)

        crop_width = xmax - xmin
        crop_height = ymax - ymin
        print(f"      → Crop region: ({xmin}, {ymin}) to ({xmax}, {ymax})")
        print(f"      → Cropped size: {crop_width}x{crop_height} pixels")

        # Crop image
        cropped = img[ymin:ymax, xmin:xmax]

        if cropped.size == 0:
            raise ValueError("Cropped image is empty - check bounding box coordinates")

        elapsed = time.time() - start_time
        print(f"      ✓ Signature cropped in {elapsed:.2f}s")

        return cropped

    def save_debug_image(self, image: np.ndarray, stage: str, emp_num: Optional[str] = None):
        """
        Save debug image for a processing stage

        Args:
            image: Image to save
            stage: Processing stage name
            emp_num: Employee number for filename
        """
        if not self.debug:
            return

        try:
            filename = f"debug_{emp_num or 'unknown'}_{stage}.jpg"
            debug_path = self.debug_folder / filename
            cv2.imwrite(str(debug_path), image)
            print(f"      💾 Debug image saved: {debug_path.name}")
        except Exception as e:
            print(f"      ⚠ Could not save debug image: {e}")

    def clean_signature(self, image: np.ndarray, step_num: int = 4, total_steps: int = 5) -> np.ndarray:
        """
        Clean signature by removing background using configured method

        Args:
            image: Input image as numpy array
            step_num: Current step number
            total_steps: Total number of steps

        Returns:
            Cleaned image with white background
        """
        self.print_step(step_num, total_steps, f"Cleaning signature background (method: {CLEANING_METHOD})...")
        start_time = time.time()

        if CLEANING_METHOD == "none":
            # No cleaning, just return original
            print(f"      → No cleaning applied (preserving original quality)")
            elapsed = time.time() - start_time
            print(f"      ✓ Skipped cleaning in {elapsed:.2f}s")
            return image

        elif CLEANING_METHOD == "color":
            # Remove background while preserving ink colors
            if TRANSPARENT_BACKGROUND:
                print(f"      → Removing background and creating transparency...")
            else:
                print(f"      → Removing background while preserving ink colors...")

            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Use Otsu's thresholding to automatically determine threshold
            _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Invert mask (we want ink to be white in mask, background to be black)
            mask = cv2.bitwise_not(mask)

            # Optional: Apply morphological operations to clean up the mask
            print(f"      → Refining mask with morphological operations...")
            kernel = np.ones((2, 2), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            if TRANSPARENT_BACKGROUND:
                # Create RGBA image with transparency
                print(f"      → Creating transparent background...")

                # Convert BGR to BGRA (add alpha channel)
                bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

                # Use the mask as the alpha channel
                bgra[:, :, 3] = mask

                elapsed = time.time() - start_time
                print(f"      ✓ Transparent background created in {elapsed:.2f}s")
                return bgra
            else:
                # Create white background
                white_bg = np.ones_like(image) * 255

                # Convert mask to 3 channels
                mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                # Blend: keep original colors where mask is white, use white background elsewhere
                # Normalize mask to 0-1 range
                mask_norm = mask_3ch.astype(float) / 255.0

                # Apply mask
                result = (image.astype(float) * mask_norm + white_bg.astype(float) * (1 - mask_norm)).astype(np.uint8)

                elapsed = time.time() - start_time
                print(f"      ✓ Background removed (colors preserved) in {elapsed:.2f}s")
                return result

        else:  # "adaptive" - classic black & white method
            # Convert to grayscale
            print(f"      → Converting to grayscale...")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply adaptive thresholding
            print(f"      → Applying adaptive thresholding (blockSize=11, C=2)...")
            cleaned = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11,  # Size of pixel neighborhood
                C=2  # Constant subtracted from mean
            )

            # Apply morphological operations to remove noise
            print(f"      → Applying morphological operations to remove noise...")
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

            # Convert back to BGR for consistency
            cleaned_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

            elapsed = time.time() - start_time
            print(f"      ✓ Background cleaned (B&W) in {elapsed:.2f}s")
            return cleaned_bgr

    def create_square_layout(self, image: np.ndarray, step_num: int = 4, total_steps: int = 5) -> np.ndarray:
        """
        Create a square canvas and position the signature according to configuration

        Args:
            image: Input signature image
            step_num: Current step number
            total_steps: Total number of steps

        Returns:
            Square image with signature positioned as configured
        """
        if not MAKE_SQUARE:
            return image

        self.print_step(step_num, total_steps, f"Creating square layout ({OUTPUT_SIZE}x{OUTPUT_SIZE}, position: {SIGNATURE_POSITION})...")
        start_time = time.time()

        img_height, img_width = image.shape[:2]
        print(f"      → Original signature size: {img_width}x{img_height}")

        # Determine canvas size
        if OUTPUT_SIZE:
            canvas_size = OUTPUT_SIZE
        else:
            # Use the larger dimension as canvas size
            canvas_size = max(img_width, img_height)

        # Calculate padding in pixels
        padding = int(canvas_size * SIGNATURE_PADDING)
        available_width = canvas_size - (2 * padding)
        available_height = canvas_size - (2 * padding)

        print(f"      → Canvas size: {canvas_size}x{canvas_size}")
        print(f"      → Padding: {padding}px, Available area: {available_width}x{available_height}")

        # Scale signature to fit within available area while maintaining aspect ratio
        scale = min(available_width / img_width, available_height / img_height)

        # Don't upscale if signature is smaller
        if scale > 1:
            scale = 1

        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        if scale != 1:
            print(f"      → Scaling signature to: {new_width}x{new_height} (scale: {scale:.2f})")
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        else:
            print(f"      → Using original size (no scaling needed)")
            resized = image
            new_width = img_width
            new_height = img_height

        # Create canvas (transparent or white)
        if TRANSPARENT_BACKGROUND and image.shape[2] == 4:  # RGBA image
            # Create transparent canvas
            canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
            print(f"      → Using transparent canvas")
        else:
            # Create white canvas
            canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255
            print(f"      → Using white canvas")

        # Calculate position based on configuration
        x_offset = (canvas_size - new_width) // 2  # Center horizontally

        if SIGNATURE_POSITION == "bottom":
            # Place at bottom with custom bottom margin
            bottom_space = int(canvas_size * BOTTOM_MARGIN)
            y_offset = canvas_size - new_height - bottom_space
            print(f"      → Positioning signature at bottom (bottom margin: {bottom_space}px / {BOTTOM_MARGIN*100:.1f}%)")
        elif SIGNATURE_POSITION == "top":
            # Place at top with padding
            y_offset = padding
            print(f"      → Positioning signature at top")
        else:  # center
            # Center vertically
            y_offset = (canvas_size - new_height) // 2
            print(f"      → Centering signature")

        # Ensure offsets are within bounds
        y_offset = max(0, min(y_offset, canvas_size - new_height))
        x_offset = max(0, min(x_offset, canvas_size - new_width))

        print(f"      → Placing at position: ({x_offset}, {y_offset})")

        # Place signature on canvas
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized

        elapsed = time.time() - start_time
        print(f"      ✓ Square layout created in {elapsed:.2f}s")

        return canvas

    def process_pdf(self, pdf_path: Path, file_num: int = 1, total_files: int = 1) -> bool:
        """
        Process a single PDF file to extract signature

        Args:
            pdf_path: Path to PDF file
            file_num: Current file number
            total_files: Total number of files

        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"\n{'='*70}")
            print(f"📄 FILE {file_num}/{total_files}: {pdf_path.name}")
            print(f"{'='*70}")

            overall_start_time = time.time()
            total_steps = 6 if MAKE_SQUARE else 5

            # Step 0: Extract employee number from filename
            print(f"\n  {datetime.now().strftime('%H:%M:%S')} [0/{total_steps}] Extracting employee number from filename...")
            employee_number = self.extract_employee_number(pdf_path.stem)
            if employee_number:
                print(f"      ✓ Found employee number: {employee_number}")
            else:
                print(f"      ⚠ Could not extract employee number from filename")

            # Step 1: Convert PDF to image
            temp_image = self.pdf_to_image(pdf_path, step_num=1, total_steps=total_steps)

            # Step 2: Get signature bounding box from OpenAI
            result = self.get_signature_bbox_from_openai(temp_image, employee_number,
                                                         step_num=2, total_steps=total_steps)
            bbox = result['signature_bbox']
            detected_employee_number = result.get('employee_number')

            # Use detected employee number if filename didn't have one
            final_employee_number = employee_number or detected_employee_number

            if not final_employee_number:
                print(f"      ⚠ No employee number found. Using PDF filename as fallback.")
                final_employee_number = pdf_path.stem

            print(f"\n      📋 Final employee number: {final_employee_number}")

            # Step 3: Crop signature
            cropped_signature = self.crop_signature(temp_image, bbox, step_num=3, total_steps=total_steps)

            # Save debug image of cropped signature before cleaning
            if self.debug:
                self.save_debug_image(cropped_signature, "2_cropped", final_employee_number)

            # Step 4: Clean signature
            cleaned_signature = self.clean_signature(cropped_signature, step_num=4, total_steps=total_steps)

            # Save debug image of cleaned signature
            if self.debug:
                self.save_debug_image(cleaned_signature, "3_cleaned", final_employee_number)

            # Step 5: Create square layout (if enabled)
            if MAKE_SQUARE:
                final_signature = self.create_square_layout(cleaned_signature, step_num=5, total_steps=total_steps)

                # Save debug image of square layout
                if self.debug:
                    self.save_debug_image(final_signature, "4_square", final_employee_number)
            else:
                final_signature = cleaned_signature

            # Step 6 (or 5 if no square): Save final signature
            save_step = 6 if MAKE_SQUARE else 5
            self.print_step(save_step, total_steps, "Saving signature to file...")
            save_start = time.time()

            # Determine output format and extension
            if OUTPUT_FORMAT == "png":
                output_path = self.output_folder / f"{final_employee_number}.png"
                # PNG compression level: 0 (no compression) to 9 (max compression)
                # For highest quality, use 0 or 1
                compression = 9 - min(9, int(OUTPUT_QUALITY / 11))  # Convert 0-100 to 9-0
                cv2.imwrite(str(output_path), final_signature, [cv2.IMWRITE_PNG_COMPRESSION, compression])
            else:  # jpg
                output_path = self.output_folder / f"{final_employee_number}.jpg"
                cv2.imwrite(str(output_path), final_signature, [cv2.IMWRITE_JPEG_QUALITY, OUTPUT_QUALITY])

            save_elapsed = time.time() - save_start
            output_size = output_path.stat().st_size / 1024  # KB

            # Get final dimensions
            final_height, final_width = final_signature.shape[:2]
            channels = final_signature.shape[2] if len(final_signature.shape) > 2 else 1
            alpha_info = " with transparency" if channels == 4 else ""
            print(f"      ✓ Saved to: {output_path.name} ({final_width}x{final_height}{alpha_info}, {output_size:.1f} KB) in {save_elapsed:.2f}s")

            # Cleanup temp file
            if temp_image.exists():
                temp_image.unlink()

            total_elapsed = time.time() - overall_start_time
            print(f"\n{'─'*70}")
            print(f"✓ SUCCESS: Processed {pdf_path.name} in {total_elapsed:.2f}s")
            if self.debug:
                print(f"  📁 Debug images saved in: {self.debug_folder.absolute()}")
            print(f"{'─'*70}")

            return True

        except Exception as e:
            print(f"\n{'─'*70}")
            print(f"✗ ERROR: Failed to process {pdf_path.name}")
            print(f"  Error: {e}")
            print(f"{'─'*70}")
            import traceback
            traceback.print_exc()
            return False

    def process_all_pdfs(self):
        """Process all PDF files in the input folder"""
        batch_start_time = time.time()

        # Find all PDF files
        pdf_files = list(self.input_folder.glob("*.pdf"))

        if not pdf_files:
            print(f"\n⚠ No PDF files found in {self.input_folder}")
            print(f"  Please add PDF files to the input folder and try again.")
            return

        print(f"\n{'═'*70}")
        print(f"📂 BATCH PROCESSING")
        print(f"{'═'*70}")
        print(f"Input folder:  {self.input_folder.absolute()}")
        print(f"Output folder: {self.output_folder.absolute()}")
        print(f"Total files:   {len(pdf_files)}")
        print(f"{'═'*70}")

        # List all files to be processed
        print(f"\n📋 Files to process:")
        for i, pdf_path in enumerate(pdf_files, 1):
            file_size = pdf_path.stat().st_size / 1024  # KB
            print(f"  {i}. {pdf_path.name} ({file_size:.1f} KB)")

        # Process each PDF
        successful = 0
        failed = 0
        failed_files = []

        print(f"\n{'═'*70}")
        print(f"🚀 STARTING EXTRACTION")
        print(f"{'═'*70}")

        for i, pdf_path in enumerate(pdf_files, 1):
            if self.process_pdf(pdf_path, file_num=i, total_files=len(pdf_files)):
                successful += 1
            else:
                failed += 1
                failed_files.append(pdf_path.name)

            # Show overall progress
            if len(pdf_files) > 1:
                print()
                self.print_progress_bar(i, len(pdf_files), prefix="Overall Progress")

        batch_elapsed = time.time() - batch_start_time

        # Summary
        print(f"\n{'═'*70}")
        print(f"📊 FINAL SUMMARY")
        print(f"{'═'*70}")
        print(f"Total PDFs:    {len(pdf_files)}")
        print(f"✓ Successful:  {successful}")
        print(f"✗ Failed:      {failed}")
        print(f"⏱ Total time:  {batch_elapsed:.2f}s")
        if len(pdf_files) > 0:
            avg_time = batch_elapsed / len(pdf_files)
            print(f"⌀ Avg per PDF: {avg_time:.2f}s")
        print(f"{'─'*70}")
        print(f"📁 Output:     {self.output_folder.absolute()}")

        if failed_files:
            print(f"\n⚠ Failed files:")
            for filename in failed_files:
                print(f"  • {filename}")

        print(f"{'═'*70}")

        if successful == len(pdf_files):
            print(f"🎉 All signatures extracted successfully!")
        elif successful > 0:
            print(f"⚠ Partial success: {successful}/{len(pdf_files)} signatures extracted")
        else:
            print(f"❌ No signatures were extracted. Please check the errors above.")

        print(f"{'═'*70}\n")

        # Cleanup temp folder
        try:
            for temp_file in self.temp_folder.glob("*"):
                temp_file.unlink()
            self.temp_folder.rmdir()
        except:
            pass


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("  PDF SIGNATURE EXTRACTOR")
    print("  Powered by OpenAI GPT-4o Vision API")
    print("="*70)

    try:
        extractor = SignatureExtractor()
        extractor.process_all_pdfs()
    except KeyboardInterrupt:
        print("\n\n⚠ Process interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
