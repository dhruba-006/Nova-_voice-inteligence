#!/usr/bin/env python
# Test script to check coqpit imports

try:
    from coqpit_config import Coqpit
    print("SUCCESS: Imported Coqpit from coqpit_config")
    print(f"Coqpit class: {Coqpit}")
except ImportError as e:
    print(f"FAILED: Cannot import from coqpit_config: {e}")

try:
    import sys
    from coqpit_config import *
    sys.modules['coqpit'] = sys.modules['coqpit_config']
    from coqpit import Coqpit as CoqpitTest
    print("SUCCESS: Created coqpit module alias and imported Coqpit")
except Exception as e:
    print(f"FAILED: Cannot create alias: {e}")
