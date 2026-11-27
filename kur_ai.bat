@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo Once kur.bat calistir. & pause & exit /b 1)
echo Yerel AI paketleri kuruluyor (PyTorch CUDA 12.8, RTX 50xx icin). Birkac GB indirir...
.venv\Scripts\python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install diffusers transformers accelerate safetensors
.venv\Scripts\python -c "import torch; print('GPU:', torch.cuda.is_available() and torch.cuda.get_device_name(0))"
echo.
echo Bitti. AI modelleri (~10 GB) ilk "AI ile duzelt" calistirmasinda indirilir.
pause
