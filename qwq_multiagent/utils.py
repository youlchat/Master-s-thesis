import re

def extract_score(text, low=1, high=6):
    matches = re.findall(r'\b([1-9]|1[0-2])\b', text)
    for s in matches:
        val = int(s)
        if low <= val <= high:
            return val
    return 0