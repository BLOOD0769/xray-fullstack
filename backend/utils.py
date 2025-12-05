# backend/utils.py
import base64

def b64_to_bytes(b64_str):
    return base64.b64decode(b64_str)

def bytes_to_b64(b):
    return base64.b64encode(b).decode('utf-8')
