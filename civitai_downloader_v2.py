#!/usr/bin/env python
"""CivitAI Image Downloader v1.4 - Modular Version

This is the new entry point for the modular version of the downloader.
Features improved metadata extraction and better rate limiting.

Usage:
    # Tag search (finds models with tag, downloads their images)
    python civitai_downloader_v2.py --mode 3 --tags "star butterfly"
    
    # By username
    python civitai_downloader_v2.py --mode 1 --username "artist_name"
    
    # By model ID
    python civitai_downloader_v2.py --mode 2 --model_id "123456"
    
    # Interactive mode
    python civitai_downloader_v2.py

Tag Search (Mode 3):
    - Searches for MODELS that are tagged with your tag
    - Downloads images from those models
    - Optional prompt check ensures images actually use the tag
    - Use --disable_prompt_check y to skip prompt verification
"""

from civitai_downloader.runner import main

if __name__ == "__main__":
    main()


