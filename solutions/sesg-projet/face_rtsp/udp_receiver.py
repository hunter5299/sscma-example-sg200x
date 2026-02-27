#!/usr/bin/env python3
# Deprecated wrapper for backward compatibility.
import sys
print("[deprecated] udp_receiver.py renamed to rtsp_receiver.py; invoking new script if available.")
import os
this_dir = os.path.dirname(__file__)
new = os.path.join(this_dir, "rtsp_receiver.py")
if os.path.exists(new):
    os.execvp(sys.executable, [sys.executable, new] + sys.argv[1:])
else:
    print("rtsp_receiver.py not found. Please run rtsp_receiver.py instead.")
    sys.exit(2)