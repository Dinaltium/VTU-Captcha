#!/bin/bash
# Start VTU Results Fetcher - Backend and Frontend
# Usage: ./start.sh
# Note: Run setup.sh first if not already done

echo "=========================================="
echo "Starting VTU Results Fetcher"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not detected!"
    echo "Please activate your virtual environment first:"
    echo "  source ~/tfenv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Load nvm if available (to use correct Node.js version)
if [ -s "$HOME/.nvm/nvm.sh" ]; then
    source "$HOME/.nvm/nvm.sh"
    nvm use --lts 2>/dev/null || nvm use default 2>/dev/null
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found!"
    echo ""
    echo "Please install Node.js and npm:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y nodejs npm"
    echo ""
    echo "Or use nvm:"
    echo "  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash"
    echo "  nvm install --lts"
    exit 1
fi

NODE_VERSION=$(node --version)
NPM_VERSION=$(npm --version)

# Check Node.js version (need >= 18 for Vite)
NODE_MAJOR=$(echo "$NODE_VERSION" | cut -d. -f1 | sed 's/v//')
if [ "$NODE_MAJOR" -lt 18 ]; then
    echo "⚠️  Warning: Node.js version $NODE_VERSION is too old!"
    echo "   Vite requires Node.js >= 18.0.0"
    echo "   Current: $NODE_VERSION"
    echo ""
    if [ -s "$HOME/.nvm/nvm.sh" ]; then
        echo "   Switching to nvm LTS version..."
        source "$HOME/.nvm/nvm.sh"
        nvm use --lts
        NODE_VERSION=$(node --version)
        NPM_VERSION=$(npm --version)
    else
        echo "   Please install Node.js 18+ or use nvm"
        exit 1
    fi
fi

echo "✓ Node.js version: $NODE_VERSION"
echo "✓ npm version: $NPM_VERSION"
echo ""

# Check Python dependencies first
echo "🔍 Checking Python dependencies..."
cd "$SCRIPT_DIR/backend/python"
python -c "import flask_cors" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Missing Python dependencies!"
    echo "   Installing from requirements.txt..."
    cd "$SCRIPT_DIR"
    pip install -r Scripts/requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install Python dependencies"
        echo "   Please run: pip install -r Scripts/requirements.txt"
        exit 1
    fi
    echo "✓ Python dependencies installed"
    echo ""
fi

# Start backend in background
echo "🚀 Starting Flask Backend..."
cd "$SCRIPT_DIR/backend/python"
python api.py &
BACKEND_PID=$!
echo "   Backend PID: $BACKEND_PID"
echo "   Backend URL: http://localhost:5000"
echo ""

# Wait a bit for backend to start
sleep 3

# Start frontend
echo "🚀 Starting React Frontend..."
cd "$SCRIPT_DIR/frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "⚠️  node_modules not found. Installing dependencies..."
    npm install
fi

npm run dev &
FRONTEND_PID=$!
echo "   Frontend PID: $FRONTEND_PID"
echo "   Frontend URL: http://localhost:5173"
echo ""

echo "=========================================="
echo "Both servers are running!"
echo "=========================================="
echo ""
echo "Backend:  http://localhost:5000"
echo "Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers..."
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✓ Servers stopped"
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT SIGTERM

# Wait for processes
wait

