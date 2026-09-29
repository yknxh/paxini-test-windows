@echo off
chcp 65001 >nul
cd /d %~dp0
where py >nul 2>nul && (py -3 -m venv .venv) || (python -m venv .venv)
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
echo.
echo 설치 완료. run_windows.bat 로 실행하세요.
pause
