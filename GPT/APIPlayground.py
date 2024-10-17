from revChatGPT.V3 import Chatbot
import os


os.environ['API_URL'] = "https://www.gptapi.us"
chatbot = Chatbot(api_key="sk-EOiCMBOnmLHLGc3x27307cAbE2094d8980980813Ba766aB7")
print("Chatbot: ")
prev_text = ""
complete_text = ""
for data in chatbot.ask(
        "你现在要回复我一段中文的文字，这段文字需要超过两句话。回复中必须用中文标点。",
):
    message = data
    print(message, end="", flush=True)
    if "。" in message or "！" in message or "？" in message:
        print('')
        print(complete_text)
        complete_text = ""
    else:
        complete_text += message
    prev_text = data
print()