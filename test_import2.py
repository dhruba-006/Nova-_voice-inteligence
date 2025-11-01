#!/usr/bin/env python
# Test to find where Coqpit actually is

import sys

# Try different import paths
attempts = [
    ('coqpit_config', 'Coqpit'),
    ('coqpit', 'Coqpit'),
    ('coqpit.coqpit', 'Coqpit'),
]

for module_name, class_name in attempts:
    try:
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name, None)
        if cls:
            print(f"✓ SUCCESS: Found {class_name} in {module_name}")
            print(f"  Module file: {getattr(module, '__file__', 'builtin')}")
            print(f"  Module contents: {dir(module)[:10]}...")
        else:
            print(f"✗ Module {module_name} exists but no {class_name} class")
    except ImportError as e:
        print(f"✗ Cannot import {module_name}: {e}")
    except Exception as e:
        print(f"✗ Error with {module_name}: {e}")

print("\n" + "="*50)
print("Checking site-packages for coqpit directories:")
import os
site_packages = os.path.join(sys.prefix, 'lib', 'site-packages')
print(f"Site packages: {site_packages}")
if os.path.exists(site_packages):
    coqpit_dirs = [d for d in os.listdir(site_packages) if 'coqpit' in d.lower()]
    for d in coqpit_dirs:
        full_path = os.path.join(site_packages, d)
        print(f"  - {d} (is_dir: {os.path.isdir(full_path)})")
        if os.path.isdir(full_path) and not d.endswith('.dist-info'):
            files = os.listdir(full_path) if os.path.isdir(full_path) else []
            print(f"    Contents: {files}")
