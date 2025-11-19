#!/bin/bash
# Setup script for PDF Signature Extractor

echo "========================================="
echo "PDF Signature Extractor - Setup"
echo "========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check for poppler-utils
echo ""
echo "Checking for poppler-utils..."
if command -v pdftoppm &> /dev/null; then
    echo "✓ poppler-utils is installed"
else
    echo "✗ poppler-utils is NOT installed"
    echo ""
    echo "Please install poppler-utils:"
    echo "  Ubuntu/Debian: sudo apt-get install poppler-utils"
    echo "  macOS: brew install poppler"
    echo "  Windows: Download from http://blog.alivate.com.au/poppler-windows/"
    exit 1
fi

# Create virtual environment (optional)
echo ""
read -p "Create virtual environment? (recommended) [y/N]: " create_venv

if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created and activated"
fi

# Install dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies"
    exit 1
fi

# Setup .env file
echo ""
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "IMPORTANT: Edit .env and add your OpenAI API key"
    echo "  Get your key from: https://platform.openai.com/api-keys"
else
    echo ".env file already exists"
fi

# Create output folder
echo ""
echo "Creating output folder..."
mkdir -p output
echo "✓ output folder created"

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env and add your OPENAI_API_KEY"
echo "2. Place PDF files in the ./input folder"
echo "3. Run: python extract_signatures.py"
echo ""
if [[ $create_venv =~ ^[Yy]$ ]]; then
    echo "Note: Remember to activate the virtual environment:"
    echo "  source venv/bin/activate"
    echo ""
fi
