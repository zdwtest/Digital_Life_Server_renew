import argparse
import asyncio
import os
import logging
import traceback
import json
from logging.handlers import TimedRotatingFileHandler

import librosa
import requests
import revChatGPT
import soundfile
import websockets

import GPT.tune
from utils.FlushingFileHandler import FlushingFileHandler
from ASR import ASRService
from GPT import GPTService
from TTS import TTService
from SentimentEngine import SentimentEngine

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
    parser.add_argument("--model", type=str, nargs='?', required=True)
    parser.add_argument("--stream", type=str2bool, nargs='?', required=True)
    parser.add_argument("--character", type=str, nargs='?', required=True)
    parser.add_argument("--ip", type=str, nargs='?', required=False)
    parser.add_argument("--baseUrl", type=str, nargs='?', required=False)
    parser.add_argument("--brainwash", type=str2bool, nargs='?', required=False)
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")  # Add WebSocket port argument
    return parser.parse_args()


class Server():
    def __init__(self, args):
        # ... (Initialization of ASR, GPT, TTS, SentimentEngine remains the same)

        self.port = args.port  # Use the port from arguments
        self.tmp_recv_file = 'tmp/server_received.wav'
        self.tmp_proc_file = 'tmp/server_processed.wav'
        self.char_name = {  # CORRECTLY INITIALIZE char_name HERE
            'paimon': ['TTS/models/paimon6k.json', 'TTS/models/paimon6k_390k.pth', 'character_paimon', 1],
            'yunfei': ['TTS/models/yunfeimix2.json', 'TTS/models/yunfeimix2_53k.pth', 'character_yunfei', 1.1],
            'catmaid': ['TTS/models/catmix.json', 'TTS/models/catmix_107k.pth', 'character_catmaid', 1.2]
        }
        # PARAFORMER
        self.paraformer = ASRService.ASRService('./ASR/resources/config.yaml')

        # LLM
        self.chat_gpt = GPTService.GPTService(args)

        # TTS
        self.tts = TTService.TTService(*self.char_name[args.character])

        # Sentiment Engine
        self.sentiment = SentimentEngine.SentimentEngine('SentimentEngine/models/paimon_sentiment.onnx')


    async def handler(self, websocket, char_name):  # Add char_name as parameter
        logging.info(f"New client connected: {websocket.remote_address}")
        print(dir(self))
        await websocket.send(char_name[args.character][2])
        async for message in websocket:

            try:
                data = json.loads(message)
                if data["type"] == "audio":
                    audio_data = bytes.fromhex(data["data"])
                    with open(self.tmp_recv_file, 'wb') as f:
                        f.write(audio_data)
                    logging.info('WAV file received and saved.')
                    ask_text = self.process_voice()

                elif data["type"] == "text":  # Handle text messages
                    ask_text = data["data"]
                    logging.info(f"Received text message: {ask_text}")

                    # Process the text message (e.g., send to GPT-3, etc.)
                    receive_text = f"Server received: {ask_text}"  # Example response
                    respond_text = self.chat_gpt.ask(ask_text)
                    # Send the response back to the client
                    await websocket.send(json.dumps({
                        "type": "text_receive",
                        "data": receive_text
                    }))

                    # Send the response back to the client
                    await websocket.send(json.dumps({
                        "type": "text_respond",
                        "data": respond_text
                    }))

                    if args.stream:
                        async for sentence in self.chat_gpt.ask_stream(ask_text):
                            await self.send_voice(websocket, sentence)
                        await self.notice_stream_end(websocket)
                    else:
                        resp_text = self.chat_gpt.ask(ask_text)
                        await self.send_voice(websocket, resp_text)
                        await self.notice_stream_end(websocket)

            except (json.JSONDecodeError, KeyError) as e:
                logging.error(f"Invalid message format: {e}")
                await websocket.send(json.dumps({"error": "Invalid message format"}))

            except (revChatGPT.typings.APIConnectionError,
                    revChatGPT.typings.Error, requests.exceptions.RequestException) as e:
                logging.error(e.__str__())
                await self.send_error(websocket, GPT.tune.error_reply, 1)

            except Exception as e:
                logging.error(e.__str__())
                logging.error(traceback.format_exc())
                await websocket.send(json.dumps({"error": "Internal server error"}))

    async def notice_stream_end(self, websocket):
        await websocket.send(json.dumps({"type": "stream_end"}))

    async def send_voice(self, websocket, resp_text, senti_or=None):
        self.tts.read_save(resp_text, self.tmp_proc_file, self.tts.hps.data.sampling_rate)
        with open(self.tmp_proc_file, 'rb') as f:
            sendData = f.read()

        if senti_or:
            senti = senti_or
        else:
            senti = self.sentiment.infer(resp_text)

        await websocket.send(json.dumps({
            "type": "audio_response",
            "data": "tmp/server_received.wav",
            #"data": sendData.hex(),
            "sentiment": int(senti)
        }))

        #logging.info('WAV SENT, size %i' % len(senddata))
        logging.info('音频地址已发送')

    # ... (process_voice, fill_size_wav remain the same)

    async def send_error(self, websocket, message, sentiment):
        # Synthesize the error message to speech
        self.tts.read_save(message, self.tmp_proc_file, self.tts.hps.data.sampling_rate)
        with open(self.tmp_proc_file, 'rb') as f:
            senddata = f.read()

        await websocket.send(json.dumps({
            "type": "audio_response",
            "data": senddata.hex(),
            "sentiment": sentiment
        }))
        await self.notice_stream_end(websocket)

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
        self.fill_size_wav()
        y, sr = librosa.load(self.tmp_recv_file, sr=None, mono=False)
        y_mono = librosa.to_mono(y)
        y_mono = librosa.resample(y_mono, orig_sr=sr, target_sr=16000)
        soundfile.write(self.tmp_recv_file, y_mono, 16000)
        text = self.paraformer.infer(self.tmp_recv_file)

        return text

    async def run(self):
        async with websockets.serve(lambda websocket: self.handler(websocket, self.char_name), "0.0.0.0",
                                    self.port):  # Pass char_name here
            logging.info(f"WebSocket server started on port {self.port}")
            await asyncio.Future()  # Run forever


if __name__ == '__main__':
    try:
        args = parse_args()
        s = Server(args)
        asyncio.run(s.run())
    except Exception as e:
        logging.error(e.__str__())
        logging.error(traceback.format_exc())
        raise e