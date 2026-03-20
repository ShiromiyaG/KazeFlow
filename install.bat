@echo off
setlocal enabledelayedexpansion
title KazeFlow Installer

echo Welcome to the KazeFlow Installer!
echo.

set "INSTALL_DIR=%cd%"
set "MINICONDA_DIR=%UserProfile%\Miniconda3"
set "ENV_DIR=%INSTALL_DIR%\env"
set "MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-py310_24.7.1-0-Windows-x86_64.exe"
set "CONDA_EXE=%MINICONDA_DIR%\Scripts\conda.exe"

set "startTime=%TIME%"
set "startHour=%TIME:~0,2%"
set "startMin=%TIME:~3,2%"
set "startSec=%TIME:~6,2%"
set /a startHour=1%startHour% - 100
set /a startMin=1%startMin% - 100
set /a startSec=1%startSec% - 100
set /a startTotal = startHour*3600 + startMin*60 + startSec

call :install_miniconda
call :create_conda_env
call :install_dependencies

set "endTime=%TIME%"
set "endHour=%TIME:~0,2%"
set "endMin=%TIME:~3,2%"
set "endSec=%TIME:~6,2%"
set /a endHour=1%endHour% - 100
set /a endMin=1%endMin% - 100
set /a endSec=1%endSec% - 100
set /a endTotal = endHour*3600 + endMin*60 + endSec
set /a elapsed = endTotal - startTotal
if %elapsed% lss 0 set /a elapsed += 86400
set /a hours = elapsed / 3600
set /a minutes = (elapsed %% 3600) / 60
set /a seconds = elapsed %% 60

echo.
echo Installation time: %hours%h %minutes%m %seconds%s
echo.
echo KazeFlow has been installed successfully!
echo Run start.bat to launch KazeFlow.
echo.
pause
exit /b 0

:install_miniconda
if exist "%CONDA_EXE%" (
    echo Miniconda already installed. Skipping.
    exit /b 0
)

echo Downloading Miniconda...
powershell -Command "& {Invoke-WebRequest -Uri '%MINICONDA_URL%' -OutFile 'miniconda.exe'}"
if not exist "miniconda.exe" goto :download_error

echo Installing Miniconda...
start /wait "" miniconda.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%MINICONDA_DIR%
if errorlevel 1 goto :install_error

del miniconda.exe
echo Miniconda installed.
echo.
exit /b 0

:create_conda_env
if exist "%ENV_DIR%" (
    echo Conda environment already exists. Skipping.
    exit /b 0
)

echo Creating conda environment...
call "%MINICONDA_DIR%\_conda.exe" create --no-shortcuts -y -k --prefix "%ENV_DIR%" python=3.10.18
if errorlevel 1 goto :error
echo Conda environment created.
echo.

if exist "%ENV_DIR%\python.exe" (
    echo Installing uv package installer...
    "%ENV_DIR%\python.exe" -m pip install uv
    if errorlevel 1 goto :error
    echo uv installed.
    echo.
)
exit /b 0

:install_dependencies
echo Installing dependencies...
call "%MINICONDA_DIR%\condabin\conda.bat" activate "%ENV_DIR%" || goto :error

echo Installing packages with uv...
uv pip install --upgrade setuptools || goto :error
uv pip install torch torchaudio --upgrade --index-url https://download.pytorch.org/whl/cu128 || goto :error
uv pip install -r "%INSTALL_DIR%\requirements.txt" || goto :error

call "%MINICONDA_DIR%\condabin\conda.bat" deactivate
echo Dependencies installed.
echo.
exit /b 0

:download_error
echo Download failed. Check your internet connection.
goto :error

:install_error
echo Miniconda installation failed.
goto :error

:error
echo An error occurred. Check the output above.
pause
exit /b 1
