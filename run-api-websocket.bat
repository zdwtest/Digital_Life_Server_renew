set SCRIPT_NAME=SocketServer.py
set CHATVER=3
set APIKey=sk-0c8c295cb63e4cf2b1c46362cd23adf6
set PROXY=http://127.0.0.1:7890
#set baseUrl=https://dashscope.aliyuncs.com/compatible-mode/v1
set STREAM=False
set CHARACTER=yunfei
set MODEL=gpt-3.5-turbo


.\venv\Scripts\python.exe WebsocketServer.py  --baseUrl "https://dashscope.aliyuncs.com/compatible-mode/v1" --chatVer 3 --APIKey sk-0c8c295cb63e4cf2b1c46362cd23adf6 --stream False --character yunfei --model qwen-max