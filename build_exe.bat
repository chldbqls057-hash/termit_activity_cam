@echo off
setlocal

echo ================================================
echo  Termite Monitor - Windows EXE Build Script
echo ================================================
echo This only needs to run ONCE, on a PC that has Python installed.
echo After it finishes, the .exe in the "dist" folder runs WITHOUT Python.
echo.
echo If Python is not installed yet: go to python.org, download 3.10+,
echo and during setup make sure to check "Add Python to PATH".
echo.
pause

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on this PC.
    echo Install Python from python.org, then run this file again.
    pause
    exit /b 1
)

echo.
echo [1/3] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [ERROR] pip upgrade failed. Check your internet connection.
    pause
    exit /b 1
)

echo.
echo [2/3] Installing required packages...
<<<<<<< HEAD
pip install -r requirements.txt
=======
python -m pip install -r requirements.txt
>>>>>>> e72ad628a9703fcdeca6b7b5105ee6cc00710f0b
if errorlevel 1 (
    echo [ERROR] Package install failed. Check your internet connection.
    pause
    exit /b 1
)

echo.
echo [3/3] Building the exe (this can take a few minutes)...
<<<<<<< HEAD
pyinstaller --noconfirm --onefile --windowed --name TermiteMonitor termite_monitor_app.py
=======
REM Using "python -m PyInstaller" instead of the bare "pyinstaller" command,
REM because on some PCs the Python Scripts folder is not in PATH and the
REM bare command is not found even though the package installed correctly.
python -m PyInstaller --noconfirm --onefile --windowed --name TermiteMonitor termite_monitor_app.py
>>>>>>> e72ad628a9703fcdeca6b7b5105ee6cc00710f0b
if errorlevel 1 (
    echo [ERROR] Build failed. Scroll up to see the detailed error message.
    pause
    exit /b 1
)

echo.
echo ================================================
echo  Build complete!
echo  Your program is here: dist\TermiteMonitor.exe
echo  Copy that one file anywhere - it runs without Python.
echo ================================================
pause
