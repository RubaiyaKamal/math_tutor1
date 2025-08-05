#!/bin/bash

# Install tesseract
apt-get update
apt-get install -y tesseract-ocr

# Install other Python packages
pip install -r requirements.txt
