set SCRIPT_NAME=SocketServer.py
set CHATVER=3
set APIKey=sk-xxxx
set PROXY=http://127.0.0.1:7890
#set baseUrl=https://dashscope.aliyuncs.com/compatible-mode/v1
set STREAM=False
set CHARACTER=yunfei
set MODEL=gpt-3.5-turbo


.\venv\Scripts\python.exe WebsocketServer.py  --baseUrl "https://dashscope.aliyuncs.com/compatible-mode/v1" --chatVer 3 --APIKey sk-xxxx --stream False --character yunfei --model qwen-max
