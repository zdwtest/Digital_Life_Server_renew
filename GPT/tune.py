import logging

def get_tune(character, model):
    filename = f"{character}{'35' if '3.5' in model else '4'}.txt"
    try:
        logging.info(f'chatGPT prompt: {filename}')
        with open(f'GPT/prompts/{filename}', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logging.error(f"Prompt file '{filename}' not found.")
        return "" # Return empty string if file is not found





exceed_reply = """
你问的太多了，我们的毛都被你撸秃了，你自己去准备一个API，或者一小时后再来吧。
"""

error_reply = """
你等一下，我连接不上大脑了。你是不是网有问题，或者是账号填错了？
"""