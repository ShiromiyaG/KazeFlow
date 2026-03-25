@echo off

:: --- WARNING SUPPRESSION ---
:: Hide "pkg_resources" deprecation warning
set PYTHONWARNINGS=ignore
:: Hide TensorFlow missing installation and info warnings
set TF_CPP_MIN_LOG_LEVEL=3
:: ---------------------------

:: get hostname of your computer and save it to variable host.
set host=127.0.0.1

:: use port 25565 as the tensorboard port.
set port=25565

:: the link to the local tensorboard webpage is as follows
set address="http://%host%:%port%"

:: display the address in the command prompt
echo TensorBoard Address: %address%

:: ask user to key in the saved model directory, example "C:\tmp\mnist_model"
set /p UserInputPath="Key in model saved directory: "

:: check if user provided a path
if "%UserInputPath%"=="" (
    echo No directory provided. Exiting.
    pause
    exit /b 1
)

:: start tensorboard (removed --bind_all to prevent conflicts with --host)
start "" tensorboard --logdir="%UserInputPath%" --host="%host%" --port="%port%" --samples_per_plugin images=9999,audio=9999

TIMEOUT /T 3 /NOBREAK >nul

:: use default browser to open tensorboard webpage
explorer %address%