from resemblyzer import preprocess_wav, VoiceEncoder
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pyaudio
import time

utterance_embeds = None
speakers = None
encoder = None
def trainVoice():
    print("training voice")
    global utterance_embeds, speakers,encoder
    encoder = VoiceEncoder("cuda" if torch.cuda.is_available() else "cpu")

    wav_fpaths = list(Path("./interview_bot/train").rglob("*.mp3"))
    speakers = list(map(lambda wav_fpath: wav_fpath.stem, wav_fpaths))
    wavs = np.array(list(map(preprocess_wav, tqdm(wav_fpaths, "Preprocessing wavs", len(wav_fpaths)))), dtype=object)

    utterance_embeds = np.array(list(map(encoder.embed_utterance, wavs)))
global stream
global audio
def listen_and_match(threshold=0.8):

    FORMAT = pyaudio.paInt16 
    CHANNELS = 1 
    RATE = 16000  
    CHUNK = 1024 
    audio = pyaudio.PyAudio()
    print("Listening...")
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)
    try:
        while True:
            frames = []
            for _ in range(0, int(RATE / CHUNK * 5)):  
                data = stream.read(CHUNK)
                frames.append(data)
            audio_data = b''.join(frames)
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_np.astype(np.float32) / 32768.0 
            test_wav = preprocess_wav(audio_float, source_sr=RATE)
            test_embedding = encoder.embed_utterance(test_wav)
            similarity_scores = cosine_similarity([test_embedding], utterance_embeds)[0]
            above_threshold_indices = np.where(similarity_scores > threshold)[0]
            if len(above_threshold_indices) > 0:
                matching_speakers = [speakers[i] for i in above_threshold_indices]
                print(f"Matching speakers: {matching_speakers}")
                print(f"Similarity scores: {similarity_scores[above_threshold_indices]}")
            else:
                print(f"No matching speaker found. {similarity_scores} ")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping listening.")
        stream.stop_stream()
        stream.close()
        audio.terminate()

def terminatebatch():
    stream.stop_stream()
    stream.close()
    audio.terminate()
