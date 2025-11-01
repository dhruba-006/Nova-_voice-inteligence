#!/usr/bin/env python
# Test TTS import issue

import sys
import importlib.util

# Check if both packages exist
coqpit_spec = importlib.util.find_spec("coqpit")
coqpit_config_spec = importlib.util.find_spec("coqpit_config")

print(f"coqpit found: {coqpit_spec is not None}")
print(f"coqpit_config found: {coqpit_config_spec is not None}")

# Try to bypass the check by removing coqpit temporarily
if coqpit_spec:
    print("\nTemporarily hiding coqpit from sys.modules...")
    import coqpit
    saved_coqpit = sys.modules.pop('coqpit', None)
    
    # Install a fake coqpit_config as coqpit
    try:
        print("Attempting TTS import...")
        from TTS.api import TTS
        print("✓ SUCCESS: TTS imported!")
    except ImportError as e:
        print(f"✗ FAILED: {e}")
    finally:
        # Restore
        if saved_coqpit:
            sys.modules['coqpit'] = saved_coqpit
