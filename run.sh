#!/bin/bash

# ==============================================================================
# BTP - Flood Prediction Web App Run Script
# ==============================================================================

# Define colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}             Flood Detection using ResNet50 - Launching${NC}"
echo -e "${BLUE}======================================================================${NC}"
echo

# 1. Setup paths and check virtual environment
if [ ! -d ".venv" ]; then
    echo -e "${RED}Error: Virtual environment (.venv) not found.${NC}"
    echo -e "Please run the setup script first: ${YELLOW}./setup.sh${NC}"
    exit 1
fi

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
    echo -e "${BLUE}Loading environment variables from .env...${NC}"
    export $(grep -v '^#' .env | xargs)
fi

echo -e "${BLUE}Activating Python Virtual Environment...${NC}"
source .venv/bin/activate

# 2. Check credentials & checkpoint folders
if [ -d "checkpoint-v3" ]; then
    export FLOOD_CHECKPOINT="$PWD/checkpoint-v3/best_dice.pth"
elif [ -d "checkpoints-v3" ]; then
    export FLOOD_CHECKPOINT="$PWD/checkpoints-v3/best_dice.pth"
fi

if [ -f "gee-key.json" ]; then
    export FLOOD_GEE_KEY="$PWD/gee-key.json"
fi

export FLOOD_JOB_DIR="$PWD/.jobs"
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1

# Clean up any previously running instances on our ports (8000 and 3000)
echo -e "${BLUE}Cleaning up ports 8000 and 3000...${NC}"
kill -9 $(lsof -t -i:8000) 2>/dev/null || true
kill -9 $(lsof -t -i:3000) 2>/dev/null || true

# 3. Launch Backend Server
echo -e "${BLUE}Starting FastAPI Backend on http://127.0.0.1:8000...${NC}"
python3 -m uvicorn api:app --host 127.0.0.1 --port 8000 --app-dir flood-detection-src &
BACKEND_PID=$!

# Give backend a moment to start
sleep 2

# Check if backend is still running
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}Error: Backend failed to start. Check your Python setup.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ FastAPI Backend started (PID: $BACKEND_PID).${NC}"

# 4. Launch Frontend Server
echo -e "${BLUE}Starting Next.js Frontend on http://localhost:3000...${NC}"
if [ ! -f "frontend/.env.local" ]; then
    echo -e "Creating frontend env file..."
    cp frontend/.env.local.example frontend/.env.local
fi

cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

# Give frontend a moment to start
sleep 2

# Check if frontend is still running
if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${RED}Error: Frontend failed to start. Check your Node.js setup.${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi
echo -e "${GREEN}✓ Next.js Frontend started (PID: $FRONTEND_PID).${NC}"

echo
echo -e "${BLUE}======================================================================${NC}"
echo -e "${GREEN}   🚀 App is running successfully!${NC}"
echo -e "   - Frontend (Browser): ${YELLOW}http://localhost:3000${NC}"
echo -e "   - Backend API:        ${YELLOW}http://127.0.0.1:8000${NC}"
echo -e "   - API Docs:           ${YELLOW}http://127.0.0.1:8000/docs${NC}"
echo -e "   - Press ${RED}Ctrl + C${NC} in this window to stop both servers at once."
echo -e "${BLUE}======================================================================${NC}"
echo

# 5. Handle cleanup when script is stopped (Ctrl + C)
cleanup() {
    echo
    echo -e "${YELLOW}Shutting down processes...${NC}"
    # Stop backend
    if kill -0 $BACKEND_PID 2>/dev/null; then
        echo -e "Stopping Backend (PID: $BACKEND_PID)..."
        kill -15 $BACKEND_PID 2>/dev/null || true
    fi
    # Stop frontend
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        echo -e "Stopping Frontend (PID: $FRONTEND_PID)..."
        kill -15 $FRONTEND_PID 2>/dev/null || true
    fi
    sleep 1
    # Make sure they are dead
    kill -9 $BACKEND_PID 2>/dev/null || true
    kill -9 $FRONTEND_PID 2>/dev/null || true
    echo -e "${GREEN}✓ All processes stopped. Goodbye!${NC}"
    exit 0
}

# Trap signals and call cleanup
trap cleanup SIGINT SIGTERM EXIT

# Keep script running and wait for background processes
wait $BACKEND_PID $FRONTEND_PID
