@echo off
set SCRIPT_NAME=SocketServer.py
set CHATVER=3
set APIKey=sk-xxxx
set PROXY=http://127.0.0.1:7890
#set baseUrl=https://dashscope.aliyuncs.com/compatible-mode/v1
set STREAM=False
set CHARACTER=yunfei
set MODEL=gpt-3.5-turbo


.\venv\Scripts\python.exe SocketServer.py  --baseUrl "https://api.2k2.cc:8000/v1" --chatVer 3 --APIKey 0 --stream False --character yunfei --model llama3_2-1b-instruct
