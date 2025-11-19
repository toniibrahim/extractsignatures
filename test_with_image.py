#!/usr/bin/env python3
"""
Test script to extract signatures from JPG/PNG images directly
Bypasses PDF conversion to test OpenAI signature detection
"""

import os
import re
import json
import base64
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def test_signature_detection(image_path: str):
    """Test signature detection on a single image"""

    print("\n" + "="*70)
    print("  IMAGE SIGNATURE DETECTION TEST")
    print("  Testing OpenAI GPT-4o Vision API")
    print("="*70)

    # Initialize OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\n❌ OPENAI_API_KEY not found in .env file")
        return False

    client = OpenAI(api_key=api_key)
    image_path = Path(image_path)

    if not image_path.exists():
        print(f"\n❌ Image not found: {image_path}")
        return False

    print(f"\n📄 Processing: {image_path.name}")
    print(f"   Size: {image_path.stat().st_size / 1024:.1f} KB")

    # Read and encode image
    print(f"\n[1/4] Encoding image to base64...")
    start_time = time.time()
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode('utf-8')
    print(f"      ✓ Encoded ({len(base64_image) / 1024:.1f} KB) in {time.time() - start_time:.2f}s")

    # Call OpenAI
    print(f"\n[2/4] Analyzing with OpenAI GPT-4o Vision API...")
    start_time = time.time()

    prompt = """Please analyze this image and provide the following information in JSON format:

1. Find the "Handwritten Signature Specimen" area and provide the bounding box coordinates.
2. If this is a full document page, extract the Employee Number if visible.
3. If this is just a signature image, indicate that in the response.

Return ONLY a JSON object with this exact structure (no additional text):
{
    "signature_bbox": {
        "ymin": <value between 0 and 1>,
        "xmin": <value between 0 and 1>,
        "ymax": <value between 0 and 1>,
        "xmax": <value between 0 and 1>
    },
    "employee_number": "<number or null>",
    "notes": "<any relevant observations>"
}

The bounding box coordinates should be normalized (0-1 range) where:
- ymin, xmin is the top-left corner of the signature area
- ymax, xmax is the bottom-right corner of the signature area
- Include only the handwritten signature itself, not any label text"""

    try:
        response = client.chat.completions.create(
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

        response_text = response.choices[0].message.content.strip()
        print(f"      ✓ API call completed in {time.time() - start_time:.2f}s")

        # Parse response
        print(f"\n[3/4] Parsing OpenAI response...")
        response_text = re.sub(r'```json\s*|\s*```', '', response_text).strip()
        result = json.loads(response_text)

        bbox = result['signature_bbox']
        print(f"      ✓ Bounding box: ymin={bbox['ymin']:.3f}, xmin={bbox['xmin']:.3f}, ymax={bbox['ymax']:.3f}, xmax={bbox['xmax']:.3f}")
        if result.get('employee_number'):
            print(f"      ✓ Employee number: {result['employee_number']}")
        if result.get('notes'):
            print(f"      ℹ Notes: {result['notes']}")

        # Crop and save signature
        print(f"\n[4/4] Extracting and saving signature...")
        start_time = time.time()

        img = cv2.imread(str(image_path))
        height, width = img.shape[:2]

        ymin = int(bbox['ymin'] * height)
        xmin = int(bbox['xmin'] * width)
        ymax = int(bbox['ymax'] * height)
        xmax = int(bbox['xmax'] * width)

        # Ensure within bounds
        ymin = max(0, min(ymin, height))
        xmin = max(0, min(xmin, width))
        ymax = max(0, min(ymax, height))
        xmax = max(0, min(xmax, width))

        print(f"      → Original size: {width}x{height} pixels")
        print(f"      → Crop region: ({xmin}, {ymin}) to ({xmax}, {ymax})")

        cropped = img[ymin:ymax, xmin:xmax]

        if cropped.size == 0:
            print(f"      ✗ Cropped image is empty!")
            return False

        # Clean signature
        print(f"      → Cleaning background with adaptive thresholding...")
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        cleaned = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            blockSize=11, C=2
        )

        # Remove noise
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # Save
        output_dir = Path("./output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"extracted_signature_test.jpg"
        cv2.imwrite(str(output_path), cleaned, [cv2.IMWRITE_JPEG_QUALITY, 95])

        output_size = output_path.stat().st_size / 1024
        print(f"      ✓ Saved to: {output_path} ({output_size:.1f} KB) in {time.time() - start_time:.2f}s")

        print("\n" + "="*70)
        print("✅ SUCCESS! Signature extracted and saved")
        print("="*70 + "\n")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default to test signature
        image_path = "./input/test signature.jpg"

    success = test_signature_detection(image_path)
    sys.exit(0 if success else 1)
