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
OUTPUT_QUALITY = 95  # JPEG quality (1-100, higher is better)
OUTPUT_DPI = 300     # DPI for output images
