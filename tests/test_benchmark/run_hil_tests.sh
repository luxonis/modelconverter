#!/usr/bin/env bash

set -e  # Exit immediately if a command fails

# Check if HUBAI_API_KEY was provided as first argument
if [ -z "$1" ]; then
  echo "Usage: $0 <HUBAI_API_KEY>"
  exit 1
fi

# Export HUBAI_API_KEY from input argument
export HUBAI_API_KEY="$1"

# Navigate to project directory
cd /tmp/modelconverter

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r pytest


# Run tests
python -m unittest discover -s tests -p '*_test.py'