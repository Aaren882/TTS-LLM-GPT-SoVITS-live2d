import asyncio
import json
import os
import subprocess
import time
from threading import Thread
import aiohttp
import psutil
import sounddevice as sd
import soundfile as sf
import torch

# 定义 Qwen2.5 模型的 API 和服务地址
MODEL_NAME = "qwen2.5:7b"
OLLAMA_API = "http://127.0.0.1:11434"
OLLAMA_GENERATE = OLLAMA_API + "/api/generate"
GPTSOVITS_API = "http://127.0.0.1:9880/"

# 检查服务是否已启动
def is_service_running(port):
    for conn in psutil.net_connections():
        if conn.status == psutil.CONN_LISTEN and conn.laddr.port == port:
            return True
    return False

# 启动 GPT-SoVITS 服务
def start_service():
    def run_service():
        path = r"C:\Users\User\Downloads\GPT-SoVITS-beta\GPT-SoVITS-beta0706"
        os.chdir(path)
        print("启动服务中...")
        subprocess.run([r'runtime\python.exe', 'api.py', '-g',
                        r"C:\Users\User\Downloads\GPT-SoVITS-beta\GPT-SoVITS-beta0706\GPT_SoVITS\pretrained_models\s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
                        '-s',
                        r"C:\Users\User\Downloads\GPT-SoVITS-beta\GPT-SoVITS-beta0706\GPT_SoVITS\pretrained_models\s2G488k.pth"])
        print("服务已启动")
    service_thread = Thread(target=run_service)
    service_thread.start()
    port = 9880
    max_retries = 20
    retry_count = 0
    while not is_service_running(port):
        if retry_count >= max_retries:
            print("服务启动超时")
            return False
        time.sleep(1)
        retry_count += 1
    print("服务启动成功")
    return True
    
async def start_service_async():
    await asyncio.to_thread(start_service)
    if is_service_running(9880):
        print("GPT-SoVITS 服务已成功启动")
    else:
        print("GPT-SoVITS 服务启动失败")

async def generate_text_async(prompt, model_name=MODEL_NAME):
    headers = {
        'Content-Type': 'application/json'
    }
    data = {
        "model": model_name,
        "prompt": prompt,
        "cache_path": "checkpoints"
    }
    generated_text = ""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(OLLAMA_GENERATE, headers=headers, json=data) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            response_data = json.loads(line.decode('utf-8'))
                            if "response" in response_data:
                                generated_text += response_data["response"]
                            if response_data.get("done", False):
                                break
                else:
                    print(f"请求失败，状态码: {response.status}")
                    error_message = await response.text()
                    print(f"伺服器回應的錯誤訊息: {error_message}")
                    return None
    except Exception as e:
        print(f"请求发生错误: {e}")
        return None

    return generated_text
async def text_to_speech_async(text, refer_wav_path, inp_refs, top_k=5, top_p=1, temperature=1, speed=1):
    params = {
        "refer_wav_path": refer_wav_path,
        "prompt_text": "声音能够传递情感，建立联系，甚至影响我们的情绪。",
        "prompt_language": "zh",
        "text": text,
        "text_language": "zh",
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "speed": speed,
        "inp_refs": inp_refs
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(GPTSOVITS_API, json=params) as response:
            if response.status == 200:
                file_path = 'temp.wav'
                with open(file_path, 'wb') as f:
                    f.write(await response.read())
                data, samplerate = sf.read(file_path)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                data_tensor = torch.tensor(data).to(device)
                data_numpy = data_tensor.cpu().numpy()
                sd.play(data_numpy, samplerate)
                sd.wait()
            else:
                print(f"TTS 请求失败: {response.status}, 错误信息: {await response.text()}")
def close_service():
    port = 9880
    for conn in psutil.net_connections():
        if conn.status == psutil.CONN_LISTEN and conn.laddr.port == port:
            pid = conn.pid
            try:
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=True)
                print(f"Process with PID {pid} using port {port} terminated.")
            except subprocess.CalledProcessError as e:
                print(f"Failed to terminate process with PID {pid}: {e}")
            return
    print(f"No process found using port {port}.")
async def main():
    user_question = input("请输入你的问题: ")
    
    await start_service_async()
    generated_answer = await generate_text_async(user_question)
    if generated_answer:
        print("生成的答案:", generated_answer)
        refer_wav = (r"C:\Users\User\Downloads\GPT-SoVITS-beta\GPT-SoVITS-beta0706\output\slicer_opt\vocal_yurmp2e151.wav.reformatted.wav_10.wav_0000471680_0000666240.wav")
        inp_refs = r"C:\Users\User\Downloads\GPT-SoVITS-beta\GPT-SoVITS-beta0706\TEMP\tmpb_88rqdn.wav"
        await text_to_speech_async(generated_answer, refer_wav_path=refer_wav, inp_refs=inp_refs)
    close_service()
# 執行主異步函數
asyncio.run(main())
