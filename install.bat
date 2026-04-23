@echo off
echo.
echo ============================================
echo   Nina IA - Instalador de Dependencias
echo ============================================
echo.

echo [1/5] Criando venv...
python -m venv venv
call venv\Scripts\activate

echo [2/5] Instalando PyTorch com CUDA 12.1...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo [3/5] Instalando PyAudio...
pip install pipwin
pipwin install pyaudio

echo [4/5] Instalando dependencias do projeto...
pip install -r requirements.txt

echo [5/5] Configurando ambiente...
copy .env.example .env
echo.
echo ============================================
echo   Instalacao concluida!
echo   Edite o arquivo .env com suas chaves.
echo   Execute: python main.py
echo ============================================
pause
