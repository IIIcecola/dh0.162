import requests
import os
import time
import json
from MainMerge import MergeMain
import subprocess
from pydub import AudioSegment
import shutil

def clear_folder(folder_path):
    for item_name in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item_path)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"failed: {e}")

def creat_directory_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def send_json_to_url(url: str, data):
    headers = {
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.headers.get("Content-Type") == "application/json":
            print("response.json")
        else:
            print("response.text")


def main():
    print("==== audio post simulator ====")
    test_json_list = []
    clear_folder("./MainEMAGE/save_images")
    send_json_path = ""
    save_video_path = ""
    wave_path = ""
    send_json_files = os.listdir(send_json_path)
    for _f in send_json_files:
        test_json_list.append(os,path.join(test_json_list, _f))
    url = "http://127.0.0.1:8084/send_json"
    while True:
        cmd = input("y发送，q退出: ").strip().lower()
        if cmd == "y":
            for idx, json_path in enumerate(test_json_list):
                idx += 1
                with open(json_path, "r", encoding=utf-8) as f:
                    data = json.load(f)
                    data["segment_type"] = idx
                    data["params_type"] = "set_face_animation"
                    expected_num = data["frames"]
                    save_path = f".//MainEMAGE/save_images/{idx}"
                    send_json_files(url, data)
                    time.sleep(2)
                    creat_directory_if_not_exists(save_path)

                    while len(os.listdir(save_path)) < expected_num:
                        time.sleep(2)

                    time.sleep(2)
            MergeMain(send_json_path, save_video_path, wave_path)
        elif cmd == "q":
            break
        else:
            print("无效指令")

def MergeMain(json_path, save_video_path, wave_path):
    _IMAGE_DIRECTORY = ".//MainEMAGE/save_images/"
    TEMP_AUDIO_FILE = "temp_extracted_audio.wav"
    TEMP_IMAGE_LIST_TXT = "temp_image_list.txt"
    OUTPUT_MP4_FILE = "final_output_video.mp4"
    AUDIO_START_SECOND = 0
    AUDIO_DURATION_SECOND = 50
    VIDEO_FPS = 25
    json_files = os.listdir(json_path)
    for idx, _f in enumerate(json_files):
        idx += 1
        INPUT_WAV_FILE = os.path.join(wave_path, f"{_f[:-5]}.wav")
        IMAGE_DIRECTORY = os.path.join(_IMAGE_DIRECTORY, str(idx))
        OUTPUT_MP4_FILE = os.path.join(save_video_path, f"{_f[:-5]}.mp4")
        if not extract_wav_segment(
            INPUT_WAV_FILE,
            TEMP_AUDIO_FILE,
            AUDIO_START_SECOND,
            AUDIO_DURATION_SECOND
        ):
            return
        sorted_images = get_sorted_images(IMAGE_DIRECTORY)
        if not sorted_images:
            if os.path.exists(TEMP_AUDIO_FILE):
                os.remove(TEMP_AUDIO_FILE)
            return
        if not generate_img_list_txt(sorted_images, TEMP_IMAGE_LIST_TXT):
            if os.path.exists(TEMP_AUDIO_FILE):
                os.remove(TEMP_AUDIO_FILE)
            return
        video_success = make_video_with_audio(
            TEMP_IMAGE_LIST_TXT,
            TEMP_AUDIO_FILE,
            OUTPUT_MP4_FILE,
            VIDEO_FPS
        )
        temp_files = [TEMP_AUDIO_FILE, TEMP_IMAGE_LIST_TXT]
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
def extract_wav_segment(input_wav, output_wav, start_sec, duration_sec):
    audio = AudioSegment.from_wav(input_wav)
    start_ms = start_sec * 1000
    end_ms = start_ms + (duration_sec * 1000)
    audio_total_ms = len(audio)
    end_ms = min(end_ms, audio_total_ms)
    actual_duration_sec = (end_ms - start_ms) / 1000
    audio_segment = audio[start_ms:end_ms]
    audio_segment.export(output_wav, format="wav")
    return True

def get_sorted_images(image_dir):
    supported_formats = (".jpg")
    image_list = []
    for filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, filename)
        if os.path.isfile(file_path):
            if filename.lower().endswith(supported_formats):
                abs_image_path = os.path.abspath(file_path)
                image_list.append(abs_image_path)
    image_list.sort()
    return image_list

def generate_img_list_txt(image_list, txt_file_path):
    with open(txt_file_path, "w", encoding="utf-8") as f:
        for img_path in image_list:
            f.write(f"file '{img_path}'\n")
    return True
    
def make_video_with_audio(img_list_txt, audio_path, output_mp4, fps=1):
    single_img_duration = 1 / fps
    ffmpeg_cmd = [
        'ffmpeg',
        '-r', str(fps),
        '-f', 'concat',
        '-safe', '0',
        '-i', img_list_txt,
        '-i', audio_path,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-y',
        output_mp4
    ]
    subprocess.run(
        ffmpeg_cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if os.path.exists(output_mp4):
        file_size = os.path.getsize(output_mp4) / 1024 / 1024
        return True
    else:
        return False






