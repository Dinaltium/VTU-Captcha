#!/bin/bash
# Setup script for VTU Results Fetcher on Linux/WSL
# Usage: ./setup.sh

echo "=========================================="
echo "VTU Results Fetcher - Setup Script"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found!"
    echo "Please install Python 3.10 or later"
    exit 1
fi

echo "✓ Python version: $(python3 --version)"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not detected!"
    echo ""
    echo "Please activate your virtual environment first:"
    echo "  source ~/tfenv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✓ Virtual environment: $VIRTUAL_ENV"
fi

echo ""

# Install Python dependencies
echo "📦 Installing Python dependencies..."
REQ_FILE="$SCRIPT_DIR/Scripts/requirements.txt"

if [ ! -f "$REQ_FILE" ]; then
    echo "❌ requirements.txt not found at $REQ_FILE"
    exit 1
fi

pip install --upgrade pip
pip install -r "$REQ_FILE"

if [ $? -ne 0 ]; then
    echo "❌ Failed to install Python dependencies"
    exit 1
fi

echo "✓ Python dependencies installed"
echo ""

# Verify GPU/CUDA support for TensorFlow
echo "🔍 Verifying GPU/CUDA support..."
python3 << 'EOF'
import sys
try:
    import tensorflow as tf
    print(f"✓ TensorFlow version: {tf.__version__}")
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✓ GPU devices found: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  - {gpu.name}")
        # Try to enable memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ GPU memory growth enabled")
        except Exception as e:
            print(f"⚠️  Could not set GPU memory growth: {e}")
    else:
        print("⚠️  No GPU devices detected!")
        print("   TensorFlow will run on CPU only")
        print("   Make sure CUDA and cuDNN are installed if you have an NVIDIA GPU")
    
    # Check CUDA availability
    if tf.test.is_built_with_cuda():
        print("✓ TensorFlow built with CUDA support")
    else:
        print("⚠️  TensorFlow not built with CUDA support")
        print("   This may indicate CPU-only TensorFlow installation")
except ImportError as e:
    print(f"❌ Failed to import TensorFlow: {e}")
    sys.exit(1)
except Exception as e:
    print(f"⚠️  Error checking GPU support: {e}")
EOF

if [ $? -ne 0 ]; then
    echo "⚠️  GPU verification had issues, but continuing..."
fi
echo ""

# Install Selenium-specific packages (ensure they're installed)
echo "📦 Verifying Selenium packages..."
pip install selenium webdriver-manager >/dev/null 2>&1
echo "✓ Selenium packages verified"
echo ""

# Load nvm if available
if [ -s "$HOME/.nvm/nvm.sh" ]; then
    source "$HOME/.nvm/nvm.sh"
    nvm use --lts 2>/dev/null || nvm use default 2>/dev/null
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "⚠️  Node.js not found!"
    echo ""
    echo "Installing Node.js and npm..."
    echo ""
    echo "Option 1: Install via apt (Ubuntu/Debian):"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install -y nodejs npm"
    echo ""
    echo "Option 2: Install via nvm (recommended):"
    echo "  curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash"
    echo "  source ~/.bashrc"
    echo "  nvm install --lts"
    echo ""
    read -p "Do you want to install Node.js via apt now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo apt-get update
        sudo apt-get install -y nodejs npm
        if [ $? -ne 0 ]; then
            echo "❌ Failed to install Node.js"
            exit 1
        fi
    else
        echo "⚠️  Skipping Node.js installation"
        echo "   Please install Node.js manually and run this script again"
        exit 1
    fi
fi

NODE_VERSION=$(node --version)
NPM_VERSION=$(npm --version)
NODE_MAJOR=$(echo "$NODE_VERSION" | cut -d. -f1 | sed 's/v//')

echo "✓ Node.js version: $NODE_VERSION"
echo "✓ npm version: $NPM_VERSION"

# Check if Node.js version is sufficient
if [ "$NODE_MAJOR" -lt 18 ]; then
    echo ""
    echo "⚠️  Warning: Node.js version $NODE_VERSION is too old!"
    echo "   Vite requires Node.js >= 18.0.0"
    echo "   Please upgrade Node.js or use nvm to install LTS version"
    echo ""
fi
echo ""

# Install frontend dependencies
echo "📦 Installing Frontend dependencies..."
cd "$SCRIPT_DIR/frontend"

if [ ! -f "package.json" ]; then
    echo "❌ package.json not found in frontend directory"
    exit 1
fi

npm install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install frontend dependencies"
    exit 1
fi

echo "✓ Frontend dependencies installed"
echo ""

# Check for Chrome/Chromium
echo "🔍 Checking for Chrome/Chromium..."
CHROME_FOUND=false
CHROME_VERSION=""

get_browser_version() {
    local binary="$1"
    local version=""
    if command -v "$binary" &> /dev/null; then
        version=$("$binary" --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+\.\d+' | head -1)
        if [ -z "$version" ]; then
            version=$("$binary" --product-version 2>/dev/null | head -1)
        fi
    fi
    echo "$version"
}

if command -v google-chrome &> /dev/null; then
    CHROME_FOUND=true
    CHROME_VERSION=$(get_browser_version "google-chrome")
    echo "✓ Google Chrome found: ${CHROME_VERSION:-unknown version}"
elif command -v chromium-browser &> /dev/null; then
    CHROME_FOUND=true
    CHROME_VERSION=$(get_browser_version "chromium-browser")
    echo "✓ Chromium found: ${CHROME_VERSION:-unknown version}"
elif command -v chromium &> /dev/null; then
    CHROME_FOUND=true
    CHROME_VERSION=$(get_browser_version "chromium")
    echo "✓ Chromium found: ${CHROME_VERSION:-unknown version}"
fi

if [ "$CHROME_FOUND" = false ]; then
    echo "⚠️  Chrome/Chromium not found"
    echo ""
    echo "Installing Google Chrome (direct .deb, no snap)..."
    
    # Download and install Google Chrome directly (much faster than snap)
    cd /tmp
    if wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb 2>/dev/null; then
        sudo apt-get install -y ./google-chrome-stable_current_amd64.deb 2>/dev/null
        rm -f google-chrome-stable_current_amd64.deb
        cd "$SCRIPT_DIR"
        
        if command -v google-chrome &> /dev/null; then
            CHROME_FOUND=true
            CHROME_VERSION=$(google-chrome --version 2>/dev/null | grep -oP '\d+\.\d+\.\d+\.\d+' | head -1)
            echo "✓ Google Chrome installed: $CHROME_VERSION"
        else
            echo "⚠️  Installation completed but Chrome not found in PATH"
        fi
    else
        echo "❌ Failed to download Google Chrome"
        echo "   You can install manually:"
        echo "     wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb"
        echo "     sudo apt install ./google-chrome-stable_current_amd64.deb"
    fi
    cd "$SCRIPT_DIR"
fi
echo ""

# Check and install ChromeDriver
echo "🔍 Checking for ChromeDriver..."
BACKEND_DIR="$SCRIPT_DIR/backend"
CHROMEDRIVER_LINUX="$BACKEND_DIR/chromedriver"
CHROMEDRIVER_WIN="$BACKEND_DIR/chromedriver.exe"

CHROMEDRIVER_NEEDED=true

if [ -f "$CHROMEDRIVER_LINUX" ]; then
    if [ -x "$CHROMEDRIVER_LINUX" ]; then
        echo "✓ ChromeDriver found and executable: $CHROMEDRIVER_LINUX"
        CHROMEDRIVER_NEEDED=false
    else
        echo "⚠️  ChromeDriver found but not executable"
        chmod +x "$CHROMEDRIVER_LINUX"
        echo "✓ Made executable"
        CHROMEDRIVER_NEEDED=false
    fi
elif [ -f "$CHROMEDRIVER_WIN" ]; then
    echo "⚠️  Windows ChromeDriver found (will not work on Linux)"
    rm -f "$CHROMEDRIVER_WIN"
    echo "   Removed Windows ChromeDriver"
fi

if [ "$CHROMEDRIVER_NEEDED" = true ]; then
    echo "⚠️  ChromeDriver not found"
    echo "   Downloading via webdriver-manager (Python)..."

    SCRIPT_DIR_ENV="$SCRIPT_DIR" python3 <<'EOF'
import os
import shutil
import stat
from webdriver_manager.chrome import ChromeDriverManager

script_dir = os.environ.get("SCRIPT_DIR_ENV") or os.getcwd()
backend_dir = os.path.join(script_dir, "backend")
target_path = os.path.join(backend_dir, "chromedriver")

try:
    downloaded_path = ChromeDriverManager().install()
    shutil.copy2(downloaded_path, target_path)
    os.chmod(
        target_path,
        stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR |
        stat.S_IRGRP | stat.S_IXGRP |
        stat.S_IROTH | stat.S_IXOTH
    )
    print(f"✓ ChromeDriver downloaded to {target_path}")
except Exception as e:
    print(f"ERROR: {e}")
    raise
EOF

    if [ $? -eq 0 ] && [ -f "$CHROMEDRIVER_LINUX" ]; then
        CHROMEDRIVER_NEEDED=false
    else
        echo ""
        echo "   Automatic download failed."
        echo "   Please download ChromeDriver manually:"
        echo "   1. Visit: https://googlechromelabs.github.io/chrome-for-testing/"
        echo "   2. Download matching version for Linux"
        echo "   3. Extract and place in: $BACKEND_DIR/"
        echo "   4. Make executable: chmod +x $CHROMEDRIVER_LINUX"
        echo ""
        read -p "Press Enter to continue..."
    fi
fi
echo ""

# Clean up Windows metadata files if present
echo "🧹 Cleaning up Windows metadata files..."
find "$SCRIPT_DIR/backend/models" -name "*Zone.Identifier" -type f -delete 2>/dev/null
echo "✓ Cleanup done"
echo ""

echo "=========================================="
echo "Setup completed!"
echo "=========================================="
echo ""
echo "📋 Next steps:"
echo "  1. Verify models are in backend/models/:"
echo "     ls -la backend/models/*.keras backend/models/*.json"
echo ""
echo "  2. If models are missing, copy from training:"
echo "     cp models/*.keras models/*.json backend/models/"
echo ""
echo "  3. Start application:"
echo "     ./start.sh"
echo ""

