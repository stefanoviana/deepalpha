@echo off
echo.
echo  ====================================
echo   DeepAlpha - Setup Wizard
echo  ====================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install from https://python.org/downloads
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: Install dependencies
echo [1/4] Installing dependencies...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo       Done.

:: Create .env if not exists
if not exist .env (
    echo [2/4] Setting up configuration...
    copy .env.example .env >nul

    echo.
    echo  Enter your license key (from purchase email):
    set /p LICENSE_KEY="  LICENSE_KEY: "

    echo.
    echo  Enter your Bitget API keys (create at bitget.com/account/newapi):
    set /p BITGET_API_KEY="  API Key: "
    set /p BITGET_SECRET="  Secret: "
    set /p BITGET_PASSPHRASE="  Passphrase: "

    echo.
    echo  Telegram notifications (optional, press Enter to skip):
    set /p TG_TOKEN="  Bot Token: "
    set /p TG_CHAT="  Chat ID: "

    :: Write to .env
    (
        echo LICENSE_KEY=%LICENSE_KEY%
        echo EXCHANGE=bitget
        echo BITGET_API_KEY=%BITGET_API_KEY%
        echo BITGET_SECRET=%BITGET_SECRET%
        echo BITGET_PASSPHRASE=%BITGET_PASSPHRASE%
        echo TELEGRAM_TOKEN=%TG_TOKEN%
        echo TELEGRAM_CHAT_ID=%TG_CHAT%
        echo LEVERAGE=5
        echo MAX_POSITIONS=3
    ) > .env
    echo       Config saved to .env
) else (
    echo [2/4] Config .env already exists, skipping.
)

:: Download model
echo [3/4] Downloading latest AI model...
python -c "from deepalpha import verify_license, update_model; verify_license(); update_model('1h')"

:: Launch
echo.
echo [4/4] Starting DeepAlpha...
echo.
echo  ====================================
echo   Setup complete! Bot is starting...
echo   Press Ctrl+C to stop.
echo  ====================================
echo.
python deepalpha.py
pause
