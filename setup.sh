#!/bin/bash
echo ""
echo "  ===================================="
echo "   DeepAlpha - Setup Wizard"
echo "  ===================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found."
    echo "  Mac: brew install python3"
    echo "  Linux: sudo apt install python3 python3-pip"
    exit 1
fi

# Install dependencies
echo "[1/4] Installing dependencies..."
pip3 install -r requirements.txt --quiet 2>/dev/null || pip install -r requirements.txt --quiet
echo "       Done."

# Create .env if not exists
if [ ! -f .env ]; then
    echo "[2/4] Setting up configuration..."
    cp .env.example .env

    echo ""
    read -p "  License key (from purchase email): " LICENSE_KEY
    echo ""
    echo "  Bitget API keys (create at bitget.com/account/newapi):"
    read -p "  API Key: " BITGET_API_KEY
    read -p "  Secret: " BITGET_SECRET
    read -p "  Passphrase: " BITGET_PASSPHRASE
    echo ""
    echo "  Telegram notifications (optional, press Enter to skip):"
    read -p "  Bot Token: " TG_TOKEN
    read -p "  Chat ID: " TG_CHAT

    cat > .env << EOF
LICENSE_KEY=$LICENSE_KEY
EXCHANGE=bitget
BITGET_API_KEY=$BITGET_API_KEY
BITGET_SECRET=$BITGET_SECRET
BITGET_PASSPHRASE=$BITGET_PASSPHRASE
TELEGRAM_TOKEN=$TG_TOKEN
TELEGRAM_CHAT_ID=$TG_CHAT
LEVERAGE=5
MAX_POSITIONS=3
EOF
    echo "       Config saved to .env"
else
    echo "[2/4] Config .env already exists, skipping."
fi

# Download model
echo "[3/4] Downloading latest AI model..."
python3 -c "from deepalpha import verify_license, update_model; verify_license(); update_model('1h')" 2>/dev/null

# Launch
echo ""
echo "[4/4] Starting DeepAlpha..."
echo ""
echo "  ===================================="
echo "   Setup complete! Bot is starting..."
echo "   Press Ctrl+C to stop."
echo "  ===================================="
echo ""
python3 deepalpha.py
