#!/usr/bin/env python
# Find the correct TTS class import

import sys
sys.path.insert(0, '.')
import coqpit_compat

try:
    from TTS.api import TTS
    print(f"TTS type: {type(TTS)}")
    print(f"TTS is: {TTS}")
    print(f"TTS dir: {[x for x in dir(TTS) if not x.startswith('_')][:10]}")
except Exception as e:
    print(f"Import failed: {e}")

# Try alternative imports
try:
    import TTS
    print(f"\nTTS module: {TTS}")
    print(f"TTS.api: {TTS.api if hasattr(TTS, 'api') else 'No api'}")
    if hasattr(TTS, 'api'):
        print(f"TTS.api.TTS: {TTS.api.TTS if hasattr(TTS.api, 'TTS') else 'No TTS class'}")
        if hasattr(TTS.api, 'TTS'):
            print(f"TTS.api.TTS type: {type(TTS.api.TTS)}")
except Exception as e:
    print(f"Alternative import failed: {e}")
