@echo off
set SCRIPT_NAME=SocketServer.py
set CHATVER=3
set PROXY=http://127.0.0.1:7890
set STREAM=False
set CHARACTER=yunfei
set MODEL=gpt-3.5-turbo


.\venv\Scripts\python.exe SocketServer.py --chatVer 3 --stream False --character yunfei --model gpt-3.5-turbo
