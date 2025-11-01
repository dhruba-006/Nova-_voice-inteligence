try:
    import sounddevice as sd
except Exception as e:
    sd = None
    print(
        f"⚠ sounddevice import failed: {e}\n"
        "  Install it with: python -m pip install sounddevice\n"
    )

import numpy as np
try:
    import pyttsx3
except Exception as e:
    pyttsx3 = None
    print(
        f"⚠ pyttsx3 import failed: {e}\n"
        "  Install it with: python -m pip install pyttsx3\n"
    )
from faster_whisper import WhisperModel
from scipy.io.wavfile import write, read
import queue
import datetime
# Coqui TTS may not be installed in all environments; import defensively
try:
    from TTS.api import TTS
except Exception as e:
    TTS = None
    print(
        f"⚠ TTS (Coqui TTS) import failed: {e}\n"
        "TTS-based fallback will be disabled. Using pyttsx3 for voice output.\n"
    )
import logging

import warnings
import sys, os
import contextlib

try:
    import google.generativeai as genai
except ImportError as e:
    print(
        f"❌ Failed to import google.generativeai: {e}\n\n"
        "This is usually due to a protobuf version conflict.\n"
        "Fix it by running:\n"
        "  pip uninstall protobuf -y\n"
        "  pip install protobuf==4.25.3\n"
        "  pip install --upgrade google-generativeai\n"
    )
    sys.exit(1)

# --------------------------
# Configuration
# --------------------------

# Load Gemini API key (set this before running: setx GEMINI_API_KEY "YOUR_KEY")
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError('❌ Missing GEMINI_API_KEY. Please set it using: setx GEMINI_API_KEY "YOUR_KEY"')

# Initialize Gemini
genai.configure(api_key=api_key)
model_gemini = genai.GenerativeModel("gemini-1.5-flash")

# Hide noisy logs
logging.getLogger("TTS").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

@contextlib.contextmanager
def suppress_stdout():
    """Temporarily suppress stdout and stderr (hide noisy logs)."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

# --------------------------
# Load TTS and Whisper
# --------------------------
# Initialize Coqui TTS model (if available) and Whisper model quietly.
with suppress_stdout():
    if TTS is not None:
        try:
            tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
        except Exception:
            tts_model = None
    else:
        tts_model = None

with suppress_stdout():
    try:
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
    except Exception:
        model = None

samplerate = 16000
channels = 1
recordings_queue = queue.Queue()

# --------------------------
# Voice Activity Detection
# --------------------------
def vad(audio, threshold=0.02):
    energy = np.mean(np.abs(audio))
    return energy > threshold

# --------------------------
# Audio Callback
# --------------------------
def callback(indata, frames, time_, status):
    if status:
        print(f"⚠ {status}", flush=True)
    recordings_queue.put(indata.copy())

# --------------------------
# Speech-to-Text
# --------------------------
def listen_and_transcribe():
    buffer = []
    silence_count = 0
    speaking = False

    while True:
        audio_chunk = recordings_queue.get()
        buffer.extend(audio_chunk)

        if vad(audio_chunk):
            speaking = True
            silence_count = 0
        else:
            if speaking:
                silence_count += 1

        if speaking and silence_count > 15:  # ~0.8 sec silence
            break

    audio_np = np.array(buffer, dtype=np.float32)
    if len(audio_np) == 0:
        return None

    # Save temporary audio file
    temp_file = "temp_audio.wav"
    write(temp_file, samplerate, (audio_np * 32767).astype(np.int16))

    # Transcribe using Whisper
    if model is None:
        print("⚠ Whisper model not loaded. Cannot transcribe.")
        return None

    with suppress_stdout():
        segments, _ = model.transcribe(temp_file, language="en")
        result = " ".join([seg.text for seg in segments]).strip()

    return result if result else None

# --------------------------
# English TTS
# --------------------------
if pyttsx3 is not None:
    try:
        engine = pyttsx3.init('sapi5')
    except Exception:
        try:
            engine = pyttsx3.init()
        except Exception:
            engine = None
else:
    engine = None

def speak_en(text):
    # Prefer pyttsx3 if available for low-latency TTS
    if engine is not None:
        try:
            engine.say(text)
            engine.runAndWait()
            return
        except Exception as e:
            print(f"⚠ English TTS failed (pyttsx3): {e}")

    # Fallback to Coqui TTS (tts_model) and play via sounddevice if available
    try:
        out_path = "tts_output.wav"
        with suppress_stdout():
            if tts_model is not None:
                tts_model.tts_to_file(text, out_path)
                print(f"➡ TTS audio saved to {out_path}")
            else:
                print(f"⚠ No TTS model available for fallback.")
    except Exception as e:
        print(f"⚠ Fallback English TTS failed: {e}")

# --------------------------
# AI Brain (Gemini)
# --------------------------
conversation_history = []  # 👈 fix: added this to store chat context

def ai_reply(prompt):
    global conversation_history
    try:
        # Add user prompt to conversation
        conversation_history.append({"role": "user", "parts": [prompt]})

        # Generate reply
        response = model_gemini.generate_content(conversation_history)

        # Extract text safely
        if response and response.text:
            reply = response.text.strip()
        else:
            reply = "I'm not sure how to respond to that."

        # Save assistant reply to history
        conversation_history.append({"role": "model", "parts": [reply]})

        return reply

    except Exception as e:
        print(f"⚠ Gemini API Error: {e}")
        return "Sorry, I couldn’t reach my brain right now."

# --------------------------
# Tool Reply (Real-time info)
# --------------------------
def tool_reply(user_text: str):
    """Handle real-time queries like date, time, etc."""
    user_text = user_text.lower()

    if "time" in user_text:
        now = datetime.datetime.now().strftime("%I:%M %p")
        return f"The current time is {now}."

    if "date" in user_text or "day" in user_text:
        today = datetime.datetime.now().strftime("%A, %d %B %Y")
        return f"Today is {today}."

    if "hello" in user_text or "hi" in user_text:
        return "Hello! How are you doing today?"

    return None

# --------------------------
# Main Loop
# --------------------------
def main():
    if sd is None:
        print("❌ sounddevice is not available. Please install it and re-run the script. Exiting.")
        return

    print("🎙 Nova (Gemini Edition) is ready. Speak and pause to get replies (Ctrl+C to exit).")

    with sd.InputStream(
        samplerate=samplerate,
        channels=channels,
        dtype="float32",
        callback=callback,
        blocksize=8000,
    ):
        while True:
            try:
                text = listen_and_transcribe()
                if not text:
                    continue

                print(f"📝 You said: {text}")

                # --- Wake word check ---
                if "nova" not in text.lower():
                    continue  # ignore anything without "nova"

                # Clean input
                clean_text = text.lower().replace("nova", "").strip()

                # If only "nova" was said
                if clean_text == "":
                    reply_text = "Yes, I'm listening."
                else:
                    tool_response = tool_reply(clean_text)
                    if tool_response:
                        reply_text = tool_response
                    else:
                        reply_text = ai_reply(clean_text)

                print(f"🤖 Nova (Gemini): {reply_text}")
                print("🔊 Speaking now...")
                speak_en(reply_text)

                # Flush queue
                while not recordings_queue.empty():
                    recordings_queue.get_nowait()

            except KeyboardInterrupt:
                print("\n👋 Exiting smoothly...")
                break
            except Exception as e:
                print(f"⚠ Error: {e}")
                continue

if __name__ == "__main__":
    main()
    