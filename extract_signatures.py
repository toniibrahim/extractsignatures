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

class SignatureExtractor:
    """Extract signatures from PDF documents using OpenAI Vision API"""

    def __init__(self, input_folder: str = "./input", output_folder: str = "./output"):
        """
        Initialize the signature extractor

        Args:
            input_folder: Folder containing PDF files
            output_folder: Folder to save extracted signatures
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.temp_folder = Path("./temp")

        # Create output and temp directories
        self.output_folder.mkdir(exist_ok=True)
        self.temp_folder.mkdir(exist_ok=True)

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
        self.print_step(step_num, total_steps, "Analyzing image with OpenAI GPT-4o Vision API...")
        start_time = time.time()

        # Encode image
        print(f"      → Encoding image to base64...")
        base64_image = self.encode_image(image_path)
        encoded_size = len(base64_image) / 1024  # KB
        print(f"      → Sending image to OpenAI ({encoded_size:.1f} KB)...")

        # Create prompt
        prompt = """Please analyze this document and provide the following information in JSON format:

1. Find the "Handwritten Signature Specimen" area and provide the bounding box coordinates.
2. Extract the Employee Number from the document if visible.

Return ONLY a JSON object with this exact structure (no additional text):
{
    "signature_bbox": {
        "ymin": <value between 0 and 1>,
        "xmin": <value between 0 and 1>,
        "ymax": <value between 0 and 1>,
        "xmax": <value between 0 and 1>
    },
    "employee_number": "<number or null>"
}

The bounding box coordinates should be normalized (0-1 range) where:
- ymin, xmin is the top-left corner of the signature area
- ymax, xmax is the bottom-right corner of the signature area
- Include only the handwritten signature itself, not the label text"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
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
                max_tokens=500,
                temperature=0
            )

            # Parse response
            response_text = response.choices[0].message.content.strip()

            # Remove markdown code blocks if present
            response_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()

            result = json.loads(response_text)

            # Use provided employee number if OpenAI couldn't find it
            if employee_number and not result.get('employee_number'):
                result['employee_number'] = employee_number

            elapsed = time.time() - start_time
            bbox = result['signature_bbox']
            print(f"      ✓ Received bounding box: ymin={bbox['ymin']:.3f}, xmin={bbox['xmin']:.3f}, ymax={bbox['ymax']:.3f}, xmax={bbox['xmax']:.3f}")
            if result.get('employee_number'):
                print(f"      ✓ Detected employee number: {result['employee_number']}")
            print(f"      ✓ API call completed in {elapsed:.2f}s")

            return result

        except Exception as e:
            print(f"      ✗ Error calling OpenAI API: {e}")
            raise

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

    def clean_signature(self, image: np.ndarray, step_num: int = 4, total_steps: int = 5) -> np.ndarray:
        """
        Clean signature by removing background using adaptive thresholding

        Args:
            image: Input image as numpy array
            step_num: Current step number
            total_steps: Total number of steps

        Returns:
            Cleaned image with white background
        """
        self.print_step(step_num, total_steps, "Cleaning signature background...")
        start_time = time.time()

        # Convert to grayscale
        print(f"      → Converting to grayscale...")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        # This creates a clean binary image where ink is black and background is white
        print(f"      → Applying adaptive thresholding (blockSize=11, C=2)...")
        cleaned = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,  # Size of pixel neighborhood
            C=2  # Constant subtracted from mean
        )

        # Optional: Apply morphological operations to remove noise
        print(f"      → Applying morphological operations to remove noise...")
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # Convert back to BGR for consistency
        cleaned_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

        elapsed = time.time() - start_time
        print(f"      ✓ Background cleaned in {elapsed:.2f}s")

        return cleaned_bgr

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
            total_steps = 5

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

            # Step 4: Clean signature
            cleaned_signature = self.clean_signature(cropped_signature, step_num=4, total_steps=total_steps)

            # Step 5: Save final signature
            self.print_step(5, total_steps, "Saving signature to file...")
            save_start = time.time()
            output_path = self.output_folder / f"{final_employee_number}.jpg"
            cv2.imwrite(str(output_path), cleaned_signature, [cv2.IMWRITE_JPEG_QUALITY, 95])
            save_elapsed = time.time() - save_start
            output_size = output_path.stat().st_size / 1024  # KB
            print(f"      ✓ Saved to: {output_path.name} ({output_size:.1f} KB) in {save_elapsed:.2f}s")

            # Cleanup temp file
            if temp_image.exists():
                temp_image.unlink()

            total_elapsed = time.time() - overall_start_time
            print(f"\n{'─'*70}")
            print(f"✓ SUCCESS: Processed {pdf_path.name} in {total_elapsed:.2f}s")
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
