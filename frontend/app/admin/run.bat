@echo off
title ChatJEEPT Master Launcher
echo ===================================================
echo   Starting ChatJEEPT System (Backend + Frontend)
echo ===================================================

:: 1. 백엔드(FastAPI 8000) 별도 창으로 실행
start "ChatJEEPT-Backend-8000" cmd /k "cd /d C:\Users\HOME-1\Desktop\jeept && python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000"

:: 2. 2초 대기 후 프론트엔드(Next.js 3000) 별도 창으로 실행
timeout /t 2 /nobreak >nul
start "ChatJEEPT-Frontend-3000" cmd /k "cd /d C:\Users\HOME-1\Desktop\jeept\frontend && npm run dev"

echo.
echo [OK] Both servers are running in separate windows!
echo Please open: http://localhost:3000
echo.
pause