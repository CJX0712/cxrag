#!/usr/bin/env bash
# CxRAG 一键部署（Linux / macOS）
# 干净环境：克隆仓库 → 执行本脚本 → demo 自动跑通，零手工干预。
set -e
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.lock
python -m cxrag.cli demo
