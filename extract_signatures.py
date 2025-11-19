#!/usr/bin/env python3
"""
PDF Signature Extractor
Extracts handwritten signatures from PDFs using OpenAI GPT-4o Vision API
"""

import os
import re
import json
import base64
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

    def pdf_to_image(self, pdf_path: Path) -> Path:
        """
        Convert the last page of PDF to high-quality JPEG

        Args:
            pdf_path: Path to PDF file

        Returns:
            Path to generated JPEG file
        """
        print(f"Converting {pdf_path.name} to image...")

        # Convert PDF to images (last page only)
        images = convert_from_path(
            pdf_path,
            dpi=300,  # High quality
            first_page=-1,  # Last page
            last_page=-1,
            fmt='jpeg'
        )

        if not images:
            raise ValueError(f"Could not convert PDF: {pdf_path}")

        # Save the last page as JPEG
        temp_image_path = self.temp_folder / f"{pdf_path.stem}_temp.jpg"
        images[0].save(temp_image_path, 'JPEG', quality=95)

        return temp_image_path

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

    def get_signature_bbox_from_openai(self, image_path: Path, employee_number: Optional[str] = None) -> dict:
        """
        Use OpenAI GPT-4o to detect signature bounding box

        Args:
            image_path: Path to image file
            employee_number: Known employee number (optional)

        Returns:
            Dictionary with bbox coordinates and employee number
        """
        print("Analyzing image with OpenAI GPT-4o...")

        # Encode image
        base64_image = self.encode_image(image_path)

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
            print(f"OpenAI response: {response_text}")

            # Remove markdown code blocks if present
            response_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()

            result = json.loads(response_text)

            # Use provided employee number if OpenAI couldn't find it
            if employee_number and not result.get('employee_number'):
                result['employee_number'] = employee_number

            return result

        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            raise

    def crop_signature(self, image_path: Path, bbox: dict) -> np.ndarray:
        """
        Crop signature area from image using bounding box

        Args:
            image_path: Path to image file
            bbox: Bounding box dictionary with normalized coordinates

        Returns:
            Cropped image as numpy array
        """
        print("Cropping signature area...")

        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        height, width = img.shape[:2]

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

        # Crop image
        cropped = img[ymin:ymax, xmin:xmax]

        if cropped.size == 0:
            raise ValueError("Cropped image is empty - check bounding box coordinates")

        return cropped

    def clean_signature(self, image: np.ndarray) -> np.ndarray:
        """
        Clean signature by removing background using adaptive thresholding

        Args:
            image: Input image as numpy array

        Returns:
            Cleaned image with white background
        """
        print("Cleaning signature background...")

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply adaptive thresholding
        # This creates a clean binary image where ink is black and background is white
        cleaned = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,  # Size of pixel neighborhood
            C=2  # Constant subtracted from mean
        )

        # Optional: Apply morphological operations to remove noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # Convert back to BGR for consistency
        cleaned_bgr = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)

        return cleaned_bgr

    def process_pdf(self, pdf_path: Path) -> bool:
        """
        Process a single PDF file to extract signature

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"\n{'='*60}")
            print(f"Processing: {pdf_path.name}")
            print(f"{'='*60}")

            # Extract employee number from filename
            employee_number = self.extract_employee_number(pdf_path.stem)
            if employee_number:
                print(f"Employee number from filename: {employee_number}")
            else:
                print("Warning: Could not extract employee number from filename")

            # Convert PDF to image
            temp_image = self.pdf_to_image(pdf_path)

            # Get signature bounding box from OpenAI
            result = self.get_signature_bbox_from_openai(temp_image, employee_number)
            bbox = result['signature_bbox']
            detected_employee_number = result.get('employee_number')

            # Use detected employee number if filename didn't have one
            final_employee_number = employee_number or detected_employee_number

            if not final_employee_number:
                print("Warning: No employee number found. Using PDF filename as fallback.")
                final_employee_number = pdf_path.stem

            print(f"Final employee number: {final_employee_number}")
            print(f"Signature bounding box: {bbox}")

            # Crop signature
            cropped_signature = self.crop_signature(temp_image, bbox)

            # Clean signature
            cleaned_signature = self.clean_signature(cropped_signature)

            # Save final signature
            output_path = self.output_folder / f"{final_employee_number}.jpg"
            cv2.imwrite(str(output_path), cleaned_signature, [cv2.IMWRITE_JPEG_QUALITY, 95])

            print(f"✓ Signature saved: {output_path}")

            # Cleanup temp file
            if temp_image.exists():
                temp_image.unlink()

            return True

        except Exception as e:
            print(f"✗ Error processing {pdf_path.name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def process_all_pdfs(self):
        """Process all PDF files in the input folder"""
        # Find all PDF files
        pdf_files = list(self.input_folder.glob("*.pdf"))

        if not pdf_files:
            print(f"No PDF files found in {self.input_folder}")
            return

        print(f"Found {len(pdf_files)} PDF file(s)")

        # Process each PDF
        successful = 0
        failed = 0

        for pdf_path in pdf_files:
            if self.process_pdf(pdf_path):
                successful += 1
            else:
                failed += 1

        # Summary
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total PDFs: {len(pdf_files)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Output folder: {self.output_folder.absolute()}")

        # Cleanup temp folder
        try:
            for temp_file in self.temp_folder.glob("*"):
                temp_file.unlink()
            self.temp_folder.rmdir()
        except:
            pass


def main():
    """Main entry point"""
    print("PDF Signature Extractor")
    print("Using OpenAI GPT-4o Vision API\n")

    try:
        extractor = SignatureExtractor()
        extractor.process_all_pdfs()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
