import argparse
import os
from pathlib import Path
import socket
import sys
import time
import logging
import traceback
from logging.handlers import TimedRotatingFileHandler

import librosa
import requests
import revChatGPT
import soundfile

import GPT.tune
from utils.FlushingFileHandler import FlushingFileHandler
from ASR import ASRService
from GPT import GPTService
from TTS import TTService
from SentimentEngine import SentimentEngine
from LLMS import LLMService
from TTS import TTSService
# 获取当前脚本的目录
current_dir = os.path.dirname(__file__)

# 构建相对路径
relative_checkpoint_path = os.path.join(current_dir, 'TTS', 'fishspeech', 'checkpoints', 'fish-speech-1.4')
relative_prompt_tokens = Path("tmp/fake.npy")


console_logger = logging.getLogger()
console_logger.setLevel(logging.INFO)
FORMAT = '%(asctime)s %(levelname)s %(message)s'
console_handler = console_logger.handlers[0]
console_handler.setFormatter(logging.Formatter(FORMAT))
console_logger.setLevel(logging.INFO)
file_handler = FlushingFileHandler("log.log", formatter=logging.Formatter(FORMAT))
file_handler.setFormatter(logging.Formatter(FORMAT))
file_handler.setLevel(logging.INFO)
console_logger.addHandler(file_handler)
console_logger.addHandler(console_handler)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chatVer", type=int, nargs='?', required=True)
    parser.add_argument("--APIKey", type=str, nargs='?', required=False)
    parser.add_argument("--email", type=str, nargs='?', required=False)
    parser.add_argument("--password", type=str, nargs='?', required=False)
    parser.add_argument("--accessToken", type=str, nargs='?', required=False)
    parser.add_argument("--proxy", type=str, nargs='?', required=False)
    parser.add_argument("--paid", type=str2bool, nargs='?', required=False)
    parser.add_argument("--model", type=str, nargs='?', required=False)
    parser.add_argument("--stream", type=str2bool, nargs='?', required=True)
    parser.add_argument("--character", type=str, nargs='?', required=True)
    parser.add_argument("--ip", type=str, nargs='?', required=False)
    parser.add_argument("--baseUrl", type=str, nargs='?', required=False)
    parser.add_argument("--brainwash", type=str2bool, nargs='?', required=False)
    parser.add_argument("--llms", type=str, nargs='?', required=False)  # 新增的 --llms 参数
    return parser.parse_args()


class Server():
    def __init__(self, args):
        # SERVER STUFF
        self.addr = None
        self.conn = None
        logging.info('Initializing Server...')
        self.host = socket.gethostbyname(socket.gethostname())
        self.port = 38438
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 10240000)
        self.s.bind((self.host, self.port))
        self.tmp_recv_file = 'tmp/server_received.wav'
        self.tmp_proc_file = 'tmp/server_processed.wav'

        ## hard coded character map
        self.char_name = {
            'paimon': ['TTS/models/paimon6k.json', 'TTS/models/paimon6k_390k.pth', 'character_paimon', 1],
            'yunfei': ['TTS/models/yunfeimix2.json', 'TTS/models/yunfeimix2_53k.pth', 'character_yunfei', 1.1],
            'catmaid': ['TTS/models/catmix.json', 'TTS/models/catmix_107k.pth', 'character_catmaid', 1.2]

        }

        # PARAFORMER
        self.paraformer = ASRService.ASRService('./ASR/resources/config.yaml')

        if not args.APIKey and not args.model and not args.llms:
            logging.error("No API key, model, or llms provided. Exiting...")
            sys.exit(1)  # 退出程序
        
        # CHAT GPT
        self.chat_gpt = GPTService.GPTService(args)

        # TTS
        self.tts = TTService.TTService(*self.char_name[args.character])

        # Sentiment Engine
        self.sentiment = SentimentEngine.SentimentEngine('SentimentEngine/models/paimon_sentiment.onnx')

            # 创建 TTS 服务实例
        self.tts_service = TTSService.TTService(
            checkpoint_path=relative_checkpoint_path,
            prompt_tokens_path=relative_prompt_tokens,
            num_samples=1,
        )

        # LLM Service（仅在没有 API 密钥的情况下初始化）
        self.llm_service = LLMService(args.llms) if not args.APIKey else None

    def listen(self):
        # MAIN SERVER LOOP
        while True:
            self.s.listen()
            logging.info(f"Server is listening on {self.host}:{self.port}...")
            self.conn, self.addr = self.s.accept()
            logging.info(f"Connected by {self.addr}")
            self.conn.sendall(b'%s' % self.char_name[args.character][2].encode())
            while True:
                try:
                    file = self.__receive_file()
                    with open(self.tmp_recv_file, 'wb') as f:
                        f.write(file)
                        logging.info('WAV file received and saved.')
                    ask_text = self.process_voice()
                    if args.stream:
                        for sentence in self.chat_gpt.ask_stream(ask_text):
                            self.send_voice(sentence)
                        self.notice_stream_end()
                        logging.info('Stream finished.')
                    else:
                        # 检查是否需要使用 LLMService
                        if self.llm_service:
                            llm_response = self.llm_service.generate_response(ask_text)
                        else:
                            # 如果没有 LLMService，使用聊天 GPT 响应
                            resp_text = self.chat_gpt.ask(ask_text)
                            self.send_voice(resp_text)
                            self.notice_stream_end()
                except revChatGPT.typings.APIConnectionError as e:
                    logging.error(e.__str__())
                    logging.info('API rate limit exceeded, sending: %s' % GPT.tune.exceed_reply)
                    self.send_voice(GPT.tune.exceed_reply, 2)
                    self.notice_stream_end()
                except revChatGPT.typings.Error as e:
                    logging.error(e.__str__())
                    logging.info('Something wrong with OPENAI, sending: %s' % GPT.tune.error_reply)
                    self.send_voice(GPT.tune.error_reply, 1)
                    self.notice_stream_end()
                except requests.exceptions.RequestException as e:
                    logging.error(e.__str__())
                    logging.info('Something wrong with internet, sending: %s' % GPT.tune.error_reply)
                    self.send_voice(GPT.tune.error_reply, 1)
                    self.notice_stream_end()
                except Exception as e:
                    logging.error(e.__str__())
                    logging.error(traceback.format_exc())
                    break

    def notice_stream_end(self):
        time.sleep(0.5)
        self.conn.sendall(b'stream_finished')

    def send_voice(self, resp_text, senti_or = None):
        # 使用新的 TTS 服务生成音频
        self.tts.generate_audio(resp_text)
        # 保存生成的音频到临时文件
        self.tts.save_audio(output_filename=self.tmp_proc_file)
        with open(self.tmp_proc_file, 'rb') as f:
            senddata = f.read()
        if senti_or:
            senti = senti_or
        else:
            senti = self.sentiment.infer(resp_text)
        senddata += b'?!'
        senddata += b'%i' % senti
        self.conn.sendall(senddata)
        time.sleep(0.5)
        logging.info('WAV SENT, size %i' % len(senddata))

    def __receive_file(self):
        file_data = b''
        while True:
            data = self.conn.recv(1024)
            # print(data)
            self.conn.send(b'sb')
            if data[-2:] == b'?!':
                file_data += data[0:-2]
                break
            if not data:
                # logging.info('Waiting for WAV...')
                continue
            file_data += data

        return file_data

    def fill_size_wav(self):
        with open(self.tmp_recv_file, "r+b") as f:
            # Get the size of the file
            size = os.path.getsize(self.tmp_recv_file) - 8
            # Write the size of the file to the first 4 bytes
            f.seek(4)
            f.write(size.to_bytes(4, byteorder='little'))
            f.seek(40)
            f.write((size - 28).to_bytes(4, byteorder='little'))
            f.flush()

    def process_voice(self):
        # stereo to mono
        self.fill_size_wav()
        y, sr = librosa.load(self.tmp_recv_file, sr=None, mono=False)
        y_mono = librosa.to_mono(y)
        y_mono = librosa.resample(y_mono, orig_sr=sr, target_sr=16000)
        soundfile.write(self.tmp_recv_file, y_mono, 16000)
        text = self.paraformer.infer(self.tmp_recv_file)

        return text


if __name__ == '__main__':
    try:
        args = parse_args()
        s = Server(args)
        s.listen()
    except Exception as e:
        logging.error(e.__str__())
        logging.error(traceback.format_exc())
        raise e
