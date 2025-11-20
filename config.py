"""
Configuration file for manual signature bbox overrides

If OpenAI is not detecting signatures correctly, you can manually specify
the bounding box coordinates here for specific employee numbers or apply
a default bbox for all PDFs.

Coordinates are normalized (0-1 range):
- ymin: Top of signature (e.g., 0.82)
- xmin: Left of signature (e.g., 0.15)
- ymax: Bottom of signature (e.g., 0.89)
- xmax: Right of signature (e.g., 0.68)
"""

# Manual overrides for specific employee numbers
MANUAL_BBOX_OVERRIDES = {
    # Example:
    # "3905": {
    #     "ymin": 0.82,
    #     "xmin": 0.15,
    #     "ymax": 0.89,
    #     "xmax": 0.68
    # },
}

# Default bbox to use for ALL PDFs (if set, overrides AI detection)
# Set to None to use AI detection
DEFAULT_BBOX = None

# Example - uncomment and adjust these values if AI detection is consistently wrong:
# DEFAULT_BBOX = {
#     "ymin": 0.82,
#     "xmin": 0.15,
#     "ymax": 0.89,
#     "xmax": 0.68
# }

# ============================================================================
# SIGNATURE CLEANING OPTIONS
# ============================================================================

# Set to True to preserve original ink colors (blue, black, etc.)
# Set to False for black & white signatures
PRESERVE_COLORS = True

# Background removal method:
# "none" - No cleaning, just crop (best quality, keeps everything)
# "color" - Remove white background but keep ink colors (recommended)
# "adaptive" - Black & white adaptive thresholding (classic method)
CLEANING_METHOD = "color"

# Quality settings
OUTPUT_QUALITY = 100  # JPEG quality (0-100, higher = better quality, larger file)
PNG_COMPRESSION_LEVEL = 9  # PNG compression (0-9, 0=no compression, 9=max compression)
OUTPUT_DPI = 300      # DPI for output images
OUTPUT_FORMAT = "png" # "png" for transparency, "jpg" for smaller files

# Background settings
TRANSPARENT_BACKGROUND = True  # True for transparent, False for white background

# ============================================================================
# OUTPUT LAYOUT OPTIONS
# ============================================================================

# Make output image square (1:1 aspect ratio)
MAKE_SQUARE = True

# Output size in pixels (width and height if square)
# If None, uses the cropped signature size
OUTPUT_SIZE = 800  # pixels (e.g., 800x800 for square)

# Signature position in square canvas
# "bottom" - Place signature at bottom (recommended)
# "center" - Place signature in center
# "top" - Place signature at top
SIGNATURE_POSITION = "bottom"

# Padding around signature (as fraction of canvas size)
# e.g., 0.1 means 10% padding on each side
SIGNATURE_PADDING = 0.05  # 5% padding

# Bottom margin (space below signature when SIGNATURE_POSITION = "bottom")
# Set to a smaller value to move signature closer to bottom edge
# e.g., 0.02 = 2% space below signature (very close to bottom)
# e.g., 0.05 = 5% space below signature (default)
# e.g., 0.10 = 10% space below signature (more space at bottom)
BOTTOM_MARGIN = 0.02  # 2% space below signature
