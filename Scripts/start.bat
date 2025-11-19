@echo off
echo ====================================
echo Starting VTU Results Fetcher
echo ====================================
echo.

echo Starting Flask Backend in conda env 'tfenv'...
start cmd /k "title Backend Server && call conda activate tfenv && cd ..\backend\python && python api.py"

timeout /t 3 /nobreak >nul

echo Starting React Frontend...
start cmd /k "title Frontend Server && cd ..\frontend && npm run dev"

echo.
echo ====================================
echo Both servers are starting...
echo ====================================
echo.
echo Backend: http://localhost:5000
echo Frontend: http://localhost:5173
echo.
echo Press any key to exit...
pause >nul
