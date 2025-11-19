#!/bin/bash

# Activate environment
source ./activate_env.sh

# Install dependencies
echo "Installing web demo dependencies..."
pip install -r requirements_web.txt

# Run server
echo "Starting Web Demo Server..."
echo "Please open http://localhost:8000 in your Windows browser."
python server.py
