import os
import datetime

# Fix coqpit import issue before importing TTS
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coqpit_compat

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
from faster_whisper import WhisperModel
from TTS.api import TTS as TTSModel
import logging
import warnings
import contextlib
from dotenv import load_dotenv
import google.generativeai as genai
from scipy.io.wavfile import write as wav_write, read as wav_read
from werkzeug.utils import secure_filename
import noisereduce as nr
import transformers
import torch
import torchaudio
import functools
import soundfile as sf

# --- START of NEW FIX ---
# We will replace the problematic function in the TTS library
# with our own version that uses 'soundfile' to avoid
# the torchaudio/torchcodec/FFmpeg error.

try:
    def patched_load_audio(audiopath, load_sr=None):
        """
        Patched function to load audio using soundfile.
        """
        # Use soundfile to read the audio
        audio_data, sample_rate = sf.read(audiopath, dtype='float32')

        # If a specific sample rate is requested, soundfile handles resampling
        if load_sr and sample_rate != load_sr:
             # This should not happen if sf.read(samplerate=...) is used,
             # but as a fallback, we just log it.
             # We will rely on the TTS model's resampler.
             pass

        # Convert numpy array to torch tensor
        audio_tensor = torch.tensor(audio_data).float()

        # Ensure it's 2D [channels, samples] or [batch, samples]
        # torchaudio.load returns [channels, samples]
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0) # [1, samples]
        else:
            audio_tensor = audio_tensor.T # [channels, samples]

        return audio_tensor, sample_rate

    # Import the module we need to patch
    import TTS.tts.models.xtts

    # Overwrite the original function with our patched version
    TTS.tts.models.xtts.load_audio = patched_load_audio

    logging.info("SUCCESS: Patched TTS.tts.models.xtts.load_audio to bypass torchcodec.")

except ImportError:
    logging.warning("WARNING: Could not patch TTS.tts.models.xtts.load_audio.")

# --- END of NEW FIX ---


# --------------------------
# Configuration
# --------------------------
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
OUTPUT_DIR = os.path.join(STATIC_DIR, 'outputs')
VOICE_SAMPLES_DIR = os.path.join(BASE_DIR, 'voice_cloner', 'voice_samples')
os.makedirs(OUTPUT_DIR, exist_ok=True)

load_dotenv()

# Load Gemini API Key
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError('Missing GEMINI_API_KEY environment variable')

logger.info(f'Gemini API key loaded: {api_key[:10]}...')
genai.configure(api_key=api_key)
model_gemini = genai.GenerativeModel('models/gemini-2.5-flash-preview-05-20')
logger.info('Gemini AI model initialized successfully')

# Reduce noisy logs
logging.getLogger('TTS').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=UserWarning)

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# Load models once
logger.info('Loading XTTS-v2 model for voice cloning...')
with suppress_stdout():
    tts_model = TTSModel('tts_models/multilingual/multi-dataset/xtts_v2', progress_bar=False, gpu=False)
logger.info('XTTS-v2 model loaded successfully!')

# Find user's voice sample
def get_voice_sample():
    """Get the user's voice sample for cloning"""
    import glob
    wav_files = glob.glob(os.path.join(VOICE_SAMPLES_DIR, '*.wav'))
    if not wav_files:
        logger.warning('No voice samples found! Using default voice.')
        return None
    # Use the first .wav file (or you can select the longest one)
    voice_sample = wav_files[0]
    logger.info(f'Using voice sample: {os.path.basename(voice_sample)}')
    return voice_sample

VOICE_SAMPLE_PATH = get_voice_sample()

with suppress_stdout():
    asr_model = WhisperModel('tiny', device='cpu', compute_type='int8')

SAMPLERATE = 16000

def reduce_noise(audio_path: str) -> str:
    """Apply noise reduction to audio file and return path to cleaned file"""
    try:
        # Read the audio file
        rate, data = wav_read(audio_path)
        
        # Convert to float32 if needed
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
            
        # Handle stereo - convert to mono
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        
        # Apply noise reduction
        reduced_noise = nr.reduce_noise(y=data, sr=rate, prop_decrease=0.8)
        
        # Save the cleaned audio
        cleaned_path = audio_path.replace('.wav', '_cleaned.wav')
        wav_write(cleaned_path, rate, (reduced_noise * 32767).astype(np.int16))
        
        return cleaned_path
    except Exception as e:
        logger.warning(f'Noise reduction failed: {e}, using original audio')
        return audio_path

def get_basic_response(user_text: str):
    """Fallback responses for basic queries when Gemini fails"""
    user_text_lower = user_text.lower()
    now = datetime.datetime.now()
    
    # Time queries
    if any(phrase in user_text_lower for phrase in ['what time', 'current time', 'tell me the time', "what's the time"]):
        return f'The current time is {now.strftime("%I:%M %p")}.'
    
    # Date queries
    if any(phrase in user_text_lower for phrase in ['what date', 'current date', 'today date', "what's the date", 'what day']):
        return f'Today is {now.strftime("%A, %B %d, %Y")}.'
    
    # Greetings
    if any(phrase in user_text_lower for phrase in ['hello', 'hi ', 'hey']):
        return 'Hello! How can I help you today?'
    
    return None

# --------------------------
# Flask app
# --------------------------
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)

@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')

# Utility: save numpy float32 mono audio to wav

def _save_wav(np_audio: np.ndarray, samplerate: int, filename: str) -> str:
    path = os.path.join(OUTPUT_DIR, filename)
    # Ensure float32 in range [-1, 1]
    audio = np.clip(np_audio, -1.0, 1.0)
    wav_write(path, samplerate, (audio * 32767).astype(np.int16))
    return path

# --------------------------
# STT endpoint
# --------------------------
@app.route('/api/stt', methods=['POST'])
def api_stt():
    if 'audio' not in request.files:
        return jsonify({'error': 'audio file missing'}), 400
    f = request.files['audio']
    fname = secure_filename(f.filename or f'upload_stt_{datetime.datetime.now().timestamp()}.wav')
    tmp_path = os.path.join(OUTPUT_DIR, fname)
    f.save(tmp_path)

    segments, _ = asr_model.transcribe(tmp_path, beam_size=5)
    text = ' '.join([seg.text for seg in segments]).strip()
    return jsonify({'text': text})

# --------------------------
# Chat endpoint (Gemini)
# --------------------------
@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json(force=True)
    prompt = data.get('prompt', '')
    history = data.get('history', [])  # optional list of {role, parts}

    if not isinstance(history, list):
        history = []
    # Build conversation for Gemini
    convo = []
    for turn in history:
        role = turn.get('role')
        parts = turn.get('parts', [])
        if role in ('user', 'model'):
            convo.append({'role': role, 'parts': parts})
    convo.append({'role': 'user', 'parts': [prompt]})

    try:
        resp = model_gemini.generate_content(convo)
        reply = resp.text.strip() if getattr(resp, 'text', None) else "I'm not sure how to respond to that."
        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --------------------------
# TTS endpoint -> returns URL to generated WAV
# --------------------------
@app.route('/api/tts', methods=['POST'])
def api_tts():
    data = request.get_json(force=True)
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'text missing'}), 400

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out_name = f'tts_{ts}.wav'
    out_path = os.path.join(OUTPUT_DIR, out_name)

    # Synthesize to file with voice cloning
    if VOICE_SAMPLE_PATH:
        tts_model.tts_to_file(
            text=text, 
            file_path=out_path,
            speaker_wav=VOICE_SAMPLE_PATH,
            language='en'
        )
    else:
        tts_model.tts_to_file(text=text, file_path=out_path)
    audio_url = f'/static/outputs/{out_name}'
    return jsonify({'audio_url': audio_url})

# --------------------------
# Combined voice chat: audio -> transcript -> tool/Gemini -> TTS
# --------------------------
@app.route('/api/voicechat', methods=['POST'])
def api_voicechat():
    wake_word = request.form.get('wake_word', 'nova').lower()
    if 'audio' not in request.files:
        return jsonify({'error': 'audio file missing'}), 400
    f = request.files['audio']

    ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S_%f')
    orig = secure_filename(f.filename or f'vc_{ts}_in.wav')
    in_path = os.path.join(OUTPUT_DIR, orig)
    f.save(in_path)

    # Apply noise reduction
    cleaned_path = reduce_noise(in_path)
    
    # Transcribe
    segments, _ = asr_model.transcribe(cleaned_path, beam_size=5)
    transcript = ' '.join([seg.text for seg in segments]).strip()

    if not transcript:
        return jsonify({'transcript': '', 'reply': "I didn't catch that.", 'audio_url': None})

    # Wake word filter
    if wake_word and wake_word not in transcript.lower():
        return jsonify({'transcript': transcript, 'reply': 'Wake word not detected. Say the wake word to talk to me.', 'audio_url': None})

    clean_text = transcript.lower().replace(wake_word, '').strip()
    
    if not clean_text:
        reply_text = "I heard the wake word but couldn't understand your question. Please try again."
        used_tool = False
    else:
        # Use Gemini AI for all responses
        try:
            logger.info(f'Sending to Gemini: {clean_text}')
            # Use simpler API call format
            resp = model_gemini.generate_content(clean_text)
            reply_text = resp.text.strip() if getattr(resp, 'text', None) else "I'm not sure how to respond to that."
            logger.info(f'Gemini response: {reply_text}')
            used_tool = False
        except Exception as e:
            logger.error(f'Gemini AI error: {str(e)}')
            logger.error(f'Full error details:', exc_info=True)
            # Fallback to basic responses
            fallback = get_basic_response(clean_text)
            if fallback:
                reply_text = fallback
                used_tool = True
                logger.info(f'Used fallback response: {reply_text}')
            else:
                reply_text = f'Sorry, my AI brain is temporarily unavailable. Please check your internet connection.'
                used_tool = False

    # TTS with voice cloning
    out_name = f'vc_{ts}_out.wav'
    out_path = os.path.join(OUTPUT_DIR, out_name)
    if VOICE_SAMPLE_PATH:
        tts_model.tts_to_file(
            text=reply_text, 
            file_path=out_path,
            speaker_wav=VOICE_SAMPLE_PATH,
            language='en'
        )
    else:
        tts_model.tts_to_file(text=reply_text, file_path=out_path)
    audio_url = f'/static/outputs/{out_name}'

    # Clean up temporary files
    try:
        if cleaned_path != in_path and os.path.exists(cleaned_path):
            os.remove(cleaned_path)
    except:
        pass
    
    return jsonify({
        'transcript': transcript,
        'reply': reply_text,
        'audio_url': audio_url,
        'used_tool': used_tool
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
