#!/bin/bash

# ==============================================================================
# BTP - Flood Prediction Web App Setup Script
# ==============================================================================

# Exit on error
set -e

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}             Flood Detection using ResNet50 - Installation${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo

# 1. Check Python installation
echo -e "${BLUE}[1/5] Checking Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python3 is not installed or not in your PATH.${NC}"
    echo -e "Please install Python 3.10+ from python.org or Homebrew."
    exit 1
fi
python3 --version
echo -e "${GREEN}✓ Python is available.${NC}"
echo

# 2. Check Node.js and npm installation
echo -e "${BLUE}[2/5] Checking Node.js and npm...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js is not installed.${NC}"
    echo -e "Please install Node.js (LTS version recommended) from nodejs.org or Homebrew.${NC}"
    exit 1
fi
if ! command -v npm &> /dev/null; then
    echo -e "${RED}Error: npm is not installed.${NC}"
    exit 1
fi
echo -e "Node.js: $(node --version)"
echo -e "npm: $(npm --version)"
echo -e "${GREEN}✓ Node.js and npm are available.${NC}"
echo

# 3. Setting up Python Virtual Environment
echo -e "${BLUE}[3/5] Setting up Python Virtual Environment (.venv)...${NC}"
if [ ! -d ".venv" ]; then
    echo -e "Creating virtual environment..."
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created successfully.${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists.${NC}"
fi

echo -e "Activating virtual environment..."
source .venv/bin/activate

echo -e "Upgrading pip..."
pip install --upgrade pip

echo -e "Installing backend python dependencies from requirements file..."
echo -e "${YELLOW}Note: This may take a few minutes as it installs scientific & geospatial packages.${NC}"
pip install -r backend-requirements.txt

echo -e "${GREEN}✓ Python dependencies installed successfully.${NC}"
echo

# 4. Setting up Frontend environment & dependencies
echo -e "${BLUE}[4/5] Setting up Frontend dependencies...${NC}"
if [ ! -d "frontend" ]; then
    echo -e "${RED}Error: 'frontend' directory not found in the current directory.${NC}"
    exit 1
fi

# Ensure env file exists
if [ ! -f "frontend/.env.local" ]; then
    echo -e "Copying frontend configuration .env.local..."
    cp frontend/.env.local.example frontend/.env.local
    echo -e "${GREEN}✓ .env.local created.${NC}"
else
    echo -e "${GREEN}✓ frontend/.env.local already exists.${NC}"
fi

cd frontend
echo -e "Installing Node dependencies..."
npm install
cd ..
echo -e "${GREEN}✓ Frontend dependencies installed successfully.${NC}"
echo

# 5. Check model and GEE credentials
echo -e "${BLUE}[5/5] Checking model files and credentials...${NC}"
if [ -f "checkpoint-v3/best_dice.pth" ] || [ -f "checkpoints-v3/best_dice.pth" ]; then
    echo -e "${GREEN}✓ Trained model weight 'best_dice.pth' found.${NC}"
else
    echo -e "${YELLOW}⚠ Warning: Model weights file 'best_dice.pth' not found in checkpoint-v3 directory.${NC}"
    echo -e "Please ensure you have placed 'best_dice.pth' inside the 'checkpoint-v3' directory."
fi

if [ -f "gee-key.json" ]; then
    echo -e "${GREEN}✓ Google Earth Engine key 'gee-key.json' found.${NC}"
else
    echo -e "${YELLOW}⚠ Warning: Google Earth Engine key 'gee-key.json' not found.${NC}"
    echo -e "Please save your service account key as 'gee-key.json' in this folder if you want S1 image extraction.${NC}"
fi
echo

echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}             🎉 SETUP COMPLETE! 🎉${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo -e "To start both backend and frontend applications in one go, run:"
echo -e "  ${YELLOW}./run.sh${NC}"
echo -e "${BLUE}======================================================================${NC}"
