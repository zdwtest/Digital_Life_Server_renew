import argparse
from GPT.GPTService import GPTService # Import from the correct location

exceed_reply = """
你问的太多了，我们的毛都被你撸秃了，你自己去准备一个API，或者一小时后再来吧。
"""

error_reply = """
你等一下，我连接不上大脑了。你是不是网有问题，或者是账号填错了？
"""

def main():
    parser = argparse.ArgumentParser(description='ChatGPT interaction script.')
    parser.add_argument('--chatVer', type=int, default=3, help='ChatGPT version (1 or 3)')
    parser.add_argument('--character', type=str, default='default', help='Character name for prompt file')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', help='Model name')
    parser.add_argument('--accessToken', type=str, default=None, help='Access token for ChatGPT v1')
    parser.add_argument('--email', type=str, default=None, help='Email for ChatGPT v1')
    parser.add_argument('--password', type=str, default=None, help='Password for ChatGPT v1')
    parser.add_argument('--paid', action='store_true', help='Paid access for ChatGPT v1')
    parser.add_argument('--APIKey', type=str, default=None, help='API key for ChatGPT v3')
    parser.add_argument('--proxy', type=str, default=None, help='Proxy server address')
    parser.add_argument('--brainwash', action='store_true', help='Brainwash mode for ChatGPT v1')

    args = parser.parse_args()

    gpt_service = GPTService(args)

    while True:
        user_input = input("请输入你的问题(输入'exit'退出): ")
        if user_input.lower() == 'exit':
            break

        try:
            response = gpt_service.ask(user_input)
            print(response)
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()