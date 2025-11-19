# PDF Signature Extractor

Automatically extract handwritten signatures from PDF documents using OpenAI's GPT-4o Vision API and OpenCV.

## Features

- **Automated Folder Scanning**: Processes all PDFs in the `./input` folder
- **Smart Signature Detection**: Uses OpenAI GPT-4o to intelligently locate signature areas
- **Employee Number Extraction**: Automatically extracts employee numbers from filenames or document content
- **Background Removal**: Applies adaptive thresholding to create clean, white-background signatures
- **High-Quality Output**: Generates high-resolution JPEG files for each signature

## How It Works

1. **Folder Scanning**: Scans the `./input` folder for all PDF files
2. **PDF to Image**: Converts the last page of each PDF to a high-quality JPEG (300 DPI)
3. **OpenAI Intelligence**:
   - Uploads the image to OpenAI GPT-4o Vision API
   - Requests bounding box coordinates for the "Handwritten Signature Specimen"
   - Extracts employee number from document as backup
4. **Cropping**: Calculates pixel coordinates and crops the signature area
5. **Background Cleaning**:
   - Uses OpenCV's Adaptive Thresholding
   - Analyzes local pixel neighborhoods
   - Converts paper texture and shadows to white
   - Preserves ink as black
6. **Saving**: Saves the cleaned signature as `[EmployeeNumber].jpg`

## Prerequisites

- Python 3.8 or higher
- OpenAI API key with access to GPT-4o
- `poppler-utils` (for PDF to image conversion)

### Installing poppler-utils

**Ubuntu/Debian**:
```bash
sudo apt-get update
sudo apt-get install poppler-utils
```

**macOS**:
```bash
brew install poppler
```

**Windows**:
Download and install from: http://blog.alivate.com.au/poppler-windows/

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd extractsignatures
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your_actual_api_key_here
```

## Usage

1. Place your PDF files in the `./input` folder

2. Run the extractor:
```bash
python extract_signatures.py
```

3. Find extracted signatures in the `./output` folder

## File Naming Convention

The program expects PDF filenames to start with the employee number:
- `3905.pdf` → Signature saved as `3905.jpg`
- `3905 - Signature Declaration.pdf` → Signature saved as `3905.jpg`
- `Employee_1234.pdf` → Signature saved as `1234.jpg`

If no number is found in the filename, the program will:
1. Try to extract the employee number from the document using GPT-4o
2. Fall back to using the full filename (without extension)

## Folder Structure

```
extractsignatures/
├── extract_signatures.py    # Main program
├── requirements.txt          # Python dependencies
├── .env                      # Your API key (create from .env.example)
├── .env.example             # Template for .env
├── input/                   # Place PDFs here
│   └── 3905 - Signature Declaration.pdf
├── output/                  # Extracted signatures appear here
│   └── 3905.jpg
└── temp/                    # Temporary files (auto-cleaned)
```

## Configuration

You can modify the `SignatureExtractor` class parameters:

```python
extractor = SignatureExtractor(
    input_folder="./input",   # Change input folder
    output_folder="./output"  # Change output folder
)
```

## Advanced Options

### Adjusting Background Removal

In the `clean_signature()` method, you can adjust the thresholding parameters:

```python
cleaned = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    blockSize=11,  # Increase for more aggressive cleaning
    C=2            # Adjust sensitivity
)
```

### Changing Image Quality

Modify the DPI in `pdf_to_image()`:

```python
images = convert_from_path(
    pdf_path,
    dpi=300,  # Increase for higher quality (e.g., 600)
    ...
)
```

## Troubleshooting

### "OPENAI_API_KEY not found"
Make sure you've created a `.env` file with your API key.

### "Could not convert PDF"
Install `poppler-utils` (see Prerequisites section).

### "Cropped image is empty"
The OpenAI API may have returned incorrect bounding box coordinates. Check the console output for the bbox values.

### Signatures have dark backgrounds
Try adjusting the `blockSize` and `C` parameters in the `clean_signature()` method.

### Cost Considerations
Each PDF requires one GPT-4o Vision API call. Monitor your usage at: https://platform.openai.com/usage

## API Costs

The program uses OpenAI's GPT-4o model with vision capabilities. Approximate costs per PDF:
- Input: ~$0.01-0.02 per PDF (depending on image size)
- Output: Minimal (JSON response)

Check current pricing: https://openai.com/api/pricing/

## Security Notes

- Never commit your `.env` file to version control
- Keep your API key secure
- The `.gitignore` file excludes sensitive files automatically

## License

MIT License - Feel free to use and modify as needed.

## Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the OpenAI API documentation: https://platform.openai.com/docs
3. Open an issue in this repository

## Example Output

**Input**: `3905 - Signature Declaration.pdf`

**Output**: `3905.jpg` - A clean, high-contrast signature image with white background

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
