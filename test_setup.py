#!/usr/bin/env python3
"""
Test script to verify the setup and dependencies
"""

import sys
import os

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking dependencies...\n")

    dependencies = {
        'openai': 'OpenAI Python SDK',
        'cv2': 'OpenCV (opencv-python)',
        'pdf2image': 'PDF2Image',
        'dotenv': 'Python-dotenv',
        'numpy': 'NumPy',
        'PIL': 'Pillow'
    }

    missing = []

    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED")
            missing.append(name)

    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False

    print("\n✓ All dependencies installed")
    return True


def check_poppler():
    """Check if poppler is installed"""
    print("\nChecking poppler-utils...")

    try:
        from pdf2image.exceptions import PDFInfoNotInstalledError
        import subprocess

        try:
            result = subprocess.run(['pdftoppm', '-v'],
                                  capture_output=True,
                                  text=True,
                                  timeout=5)
            print(f"✓ poppler-utils installed")
            return True
        except FileNotFoundError:
            print("✗ poppler-utils NOT installed")
            print("\nInstallation instructions:")
            print("  Ubuntu/Debian: sudo apt-get install poppler-utils")
            print("  macOS: brew install poppler")
            print("  Windows: Download from http://blog.alivate.com.au/poppler-windows/")
            return False
        except subprocess.TimeoutExpired:
            print("✗ poppler-utils check timed out")
            return False

    except Exception as e:
        print(f"Error checking poppler: {e}")
        return False


def check_env_file():
    """Check if .env file exists and has API key"""
    print("\nChecking configuration...")

    if not os.path.exists('.env'):
        print("✗ .env file not found")
        print("  Create one: cp .env.example .env")
        print("  Then add your OPENAI_API_KEY")
        return False

    print("✓ .env file exists")

    # Check if API key is set
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key == 'your_openai_api_key_here':
        print("✗ OPENAI_API_KEY not set in .env file")
        print("  Get your key from: https://platform.openai.com/api-keys")
        return False

    print("✓ OPENAI_API_KEY is set")
    return True


def check_folders():
    """Check if required folders exist"""
    print("\nChecking folders...")

    folders = {
        'input': './input',
        'output': './output'
    }

    all_exist = True

    for name, path in folders.items():
        if os.path.exists(path):
            print(f"✓ {name}/ folder exists")
        else:
            print(f"✗ {name}/ folder not found")
            all_exist = False

    # Count PDFs in input folder
    if os.path.exists('./input'):
        pdf_files = [f for f in os.listdir('./input') if f.endswith('.pdf')]
        if pdf_files:
            print(f"\n  Found {len(pdf_files)} PDF file(s) in input/")
        else:
            print("\n  Warning: No PDF files found in input/")

    return all_exist


def main():
    """Main test function"""
    print("="*60)
    print("PDF Signature Extractor - Setup Verification")
    print("="*60)
    print()

    checks = [
        ("Dependencies", check_dependencies),
        ("Poppler", check_poppler),
        ("Configuration", check_env_file),
        ("Folders", check_folders)
    ]

    results = []

    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\nError checking {name}: {e}")
            results.append((name, False))
        print()

    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = all(result for _, result in results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")

    print()

    if all_passed:
        print("✓ All checks passed! You're ready to run the extractor.")
        print("\nUsage:")
        print("  python extract_signatures.py")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
