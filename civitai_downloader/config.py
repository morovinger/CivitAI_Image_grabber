"""Configuration constants and settings for CivitAI Downloader."""

import os
import sys
import logging
import argparse
from typing import Dict

# ===================
# --- Constants ---
# ===================
BASE_API_URL: str = "https://civitai.com/api/v1/images"
MODELS_API_URL: str = "https://civitai.com/api/v1/models"

DEFAULT_HEADERS: Dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Content-Type": "application/json"
}

DEFAULT_SEMAPHORE_LIMIT: int = 5
DEFAULT_OUTPUT_DIR: str = "image_downloads"
DATABASE_FILENAME: str = "tracking_database.sqlite"
LOG_FILENAME_TEMPLATE: str = "civit_image_downloader_log_{version}.txt"
SCRIPT_VERSION: str = "1.4-modular"
DEFAULT_TIMEOUT: int = 60
DEFAULT_RETRIES: int = 2
DEFAULT_MAX_PATH_LENGTH: int = 240

# Retry configuration
RETRYABLE_STATUS_CODES: set = {500, 502, 503, 504}

# Color codes for terminal output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# Directory structure
SCRIPT_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =======================
# --- Logging Setup ---
# =======================
def setup_logging() -> logging.Logger:
    """Configure and return the main logger."""
    log_file_path = os.path.join(SCRIPT_DIR, LOG_FILENAME_TEMPLATE.format(version=SCRIPT_VERSION))
    
    logger = logging.getLogger('CivitaiDownloader')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(log_formatter)
        console_handler.setFormatter(log_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger


# ===========================
# --- Argument Parser ---
# ===========================
def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CivitAI Image Downloader (Modular Version)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="HTTP request timeout (seconds).")
    parser.add_argument("--quality", type=int, choices=[1, 2],
                        help="Image quality: 1=SD, 2=HD.")
    parser.add_argument("--redownload", type=int, choices=[1, 2], default=2,
                        help="Allow re-downloading tracked images: 1=Yes, 2=No.")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3, 4, 5, 6],
                        required=not sys.stdin.isatty(),
                        help="Download mode: 1=user, 2=model ID, 3=model tag search, 4=model version ID, 5=direct image tag search, 6=website search (Meilisearch).")
    parser.add_argument("--tags", help="Tag(s) for Mode 3/5/6 (comma-separated).")
    parser.add_argument("--disable_prompt_check", choices=['y', 'n'], default='n',
                        help="Disable prompt check in tag mode (y/n).")
    parser.add_argument("--username", help="Username(s) for Mode 1 (comma-separated).")
    parser.add_argument("--model_id", help="Model ID(s) for Mode 2 (comma-separated, numeric).")
    parser.add_argument("--model_version_id",
                        help="Model Version ID(s) for Mode 4 (comma-separated, numeric).")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Base directory for downloads.")
    parser.add_argument("--semaphore_limit", type=int, default=DEFAULT_SEMAPHORE_LIMIT,
                        help="Max concurrent downloads/API calls.")
    parser.add_argument("--no_sort", action='store_true',
                        help="Disable sorting images into model subfolders.")
    parser.add_argument("--max_path", type=int, default=DEFAULT_MAX_PATH_LENGTH,
                        help="Approximate max length for file paths.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                        help="Number of retries for failures.")
    parser.add_argument("--skip_db_tracking", choices=['y', 'n'], default='y',
                        help="Skip database image tracking (y/n). Default: y (tracking disabled).")
    
    return parser.parse_args()


