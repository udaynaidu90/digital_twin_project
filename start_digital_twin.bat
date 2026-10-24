@echo off

title Industrial Digital Twin

cd /d "C:\Users\praga\Downloads\digital_twin_project\digital_twin_project"

echo ==========================================
echo       INDUSTRIAL DIGITAL TWIN
echo ==========================================
echo.
echo Starting Live Simulator...
echo.

start "Digital Twin Simulator" cmd /k ""C:\Users\praga\AppData\Local\Programs\Python\Python313\python.exe" simulate_stream.py"

timeout /t 5 /nobreak >nul

echo.
echo Starting Streamlit Dashboard...
echo.

start "Digital Twin Dashboard" cmd /k ""C:\Users\praga\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run app.py"

timeout /t 8 /nobreak >nul

echo.
echo Opening Dashboard...
echo.

start "" "http://localhost:8501"

echo.
echo ==========================================
echo Digital Twin Started Successfully!
echo ==========================================
echo.
echo Simulator and Dashboard are running.
echo Do not close their command windows.
echo.

pause

"