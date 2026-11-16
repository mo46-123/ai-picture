@echo off
chcp 65001 >nul
rem SeoulGlow one-click start for Windows. Double-click this file.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo 没有找到 Python。请先到 https://www.python.org/downloads/ 安装 Python 3.10 以上，
  echo 安装时勾选 "Add python.exe to PATH"，然后重新双击本文件。
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] 第一次运行：正在创建环境并安装依赖，需要几分钟...
  python -m venv .venv || goto :fail
  ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
)

if not exist "models\face_landmarker.task" (
  echo [2/3] 正在下载人脸模型...
  if exist "%USERPROFILE%\Downloads\face_landmarker.task" (
    copy /y "%USERPROFILE%\Downloads\face_landmarker.task" "models\face_landmarker.task" >nul
  ) else (
    ".venv\Scripts\python.exe" scripts\download_models.py
  )
)

echo [3/3] 启动中，浏览器会自动打开 http://127.0.0.1:8000
echo       用完后关掉这个黑色窗口即可停止。
start "" http://127.0.0.1:8000
".venv\Scripts\python.exe" -m seoulglow.web
pause
exit /b 0

:fail
echo 安装失败，请把上面的报错信息截图发给开发者。
pause
exit /b 1
