@echo off
REM CxRAG 一键部署（Windows）
REM 干净环境：克隆仓库 → 双击本脚本 → demo 自动跑通，零手工干预。
cd /d "%~dp0"
python -m venv .venv
call .venv\Scripts\activate.bat
pip install --upgrade pip -q
pip install -r requirements.lock
python -m cxrag.cli demo
pause
