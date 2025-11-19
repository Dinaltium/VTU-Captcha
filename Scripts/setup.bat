@echo off
echo ====================================
echo VTU Results Fetcher - Setup Script (uses conda env: tfenv)
echo ====================================
echo.

REM --- Debug instrumentation: write a small startup log to help diagnose silent exits ---
set "DEBUG_LOG=%~dp0setup_debug.log"
echo ======= setup.bat startup =======> "%DEBUG_LOG%"
echo TIMESTAMP: %date% %time% >> "%DEBUG_LOG%"
echo ARGS: %* >> "%DEBUG_LOG%"
echo CURRENT DIR: %CD% >> "%DEBUG_LOG%"
echo SHELL PID: %PROCESSOR_IDENTIFIER% >> "%DEBUG_LOG%"
set >> "%DEBUG_LOG%" 2>nul
echo (Wrote debug env to %DEBUG_LOG%)


REM Check for conda. If not present, fall back to a local .venv in the project root
conda --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 'conda' not found in PATH. Falling back to project virtualenv '.venv' (if available) or creating one.

    if exist "%~dp0..\.venv\Scripts\pip.exe" (
        echo Found existing virtualenv at %~dp0..\.venv
        echo Installing Python dependencies into .venv
        "%~dp0..\.venv\Scripts\pip.exe" install -r "%~dp0requirements.txt"
        if %errorlevel% neq 0 (
            echo Error: Failed to install Python dependencies into .venv
            pause
            exit /b 1
        )
    ) else (
        echo No existing .venv found. Attempting to create one using 'python -m venv .venv'
        python -V >nul 2>&1
        if %errorlevel% neq 0 (
            echo Error: 'python' not found in PATH. Please install Python or Anaconda/Miniconda and retry.
            pause
            exit /b 1
        )

        pushd "%~dp0.."
        python -m venv .venv
        if %errorlevel% neq 0 (
            echo Error: Failed to create virtual environment .venv
            popd
            pause
            exit /b 1
        )
        echo Installing Python dependencies into new .venv
        .venv\Scripts\pip.exe install -r "%~dp0requirements.txt"
        if %errorlevel% neq 0 (
            echo Error: Failed to install Python dependencies into .venv
            popd
            pause
            exit /b 1
        )
        popd
    )
)
REM 'conda' is available. If tfenv is already active in this shell, install directly with pip.
if defined CONDA_DEFAULT_ENV (
    if "%CONDA_DEFAULT_ENV%"=="tfenv" (
        echo Detected active conda env 'tfenv' in this shell. Installing packages with pip...
        pip install -r "%~dp0requirements.txt"
        if %errorlevel% neq 0 (
            echo Error: Failed to install Python dependencies into active 'tfenv'
            pause
            exit /b 1
        )
        goto SKIP_CONDA_SETUP
    )
)

echo Checking for conda environment 'tfenv'...
conda env list | findstr /R /C:"\btfenv\b" >nul 2>&1
if %errorlevel% neq 0 (
    echo Creating conda environment 'tfenv' with Python 3.9...
    conda create -y -n tfenv python=3.9
    if %errorlevel% neq 0 (
        echo Error: Failed to create conda environment 'tfenv'
        pause
        exit /b 1
    )
) else (
    echo Found existing 'tfenv' environment.
)

echo Installing Python dependencies into 'tfenv'...
REM Use conda run so we don't rely on shell activation inside this script
conda run -n tfenv pip install -r "%~dp0requirements.txt"
if %errorlevel% neq 0 (
    echo Error: Failed to install Python dependencies into 'tfenv'
    pause
    exit /b 1
)

:SKIP_CONDA_SETUP
else (
    echo Checking for conda environment 'tfenv'...

    REM If the current shell already has tfenv activated, install directly with pip
    if defined CONDA_DEFAULT_ENV (
        if "%CONDA_DEFAULT_ENV%"=="tfenv" (
            echo Detected active conda env 'tfenv' in this shell. Installing packages with pip...
            pip install -r "%~dp0requirements.txt"
            if %errorlevel% neq 0 (
                echo Error: Failed to install Python dependencies into active 'tfenv'
                pause
                exit /b 1
            )
            goto AFTER_PIP_INSTALL
        )
    )

    conda env list | findstr /R /C:"\btfenv\b" >nul 2>&1
    if %errorlevel% neq 0 (
        echo Creating conda environment 'tfenv' with Python 3.9...
        conda create -y -n tfenv python=3.9
        if %errorlevel% neq 0 (
            echo Error: Failed to create conda environment 'tfenv'
            pause
            exit /b 1
        )
    ) else (
        echo Found existing 'tfenv' environment.
    )

    echo Installing Python dependencies into 'tfenv'...
    REM Use conda run so we don't rely on shell activation inside this script
    conda run -n tfenv pip install -r "%~dp0requirements.txt"
    if %errorlevel% neq 0 (
        echo Error: Failed to install Python dependencies into 'tfenv'
        pause
        exit /b 1
    )

    :AFTER_PIP_INSTALL
)

echo.
echo Installing Frontend dependencies...
cd ..\frontend
call npm install

if %errorlevel% neq 0 (
    echo Error: Failed to install frontend dependencies
    pause
    exit /b 1
)

cd ..\Scripts

echo.
echo ====================================
echo Setup completed successfully!
echo ====================================
echo.
echo To run the application:
echo 1. Start backend:  call conda activate tfenv ^&^& python ..\backend\python\api.py
echo 2. Start frontend: cd ..\frontend ^&^& npm run dev
echo.
echo Press any key to exit...
pause >nul
