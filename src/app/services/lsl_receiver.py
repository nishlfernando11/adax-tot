from pylsl import StreamInlet, resolve_byprop, proc_ALL
import time
import pylsl
import json


def resolve_streams():
    streams = pylsl.resolve_streams()
    print(f"Found {len(streams)} streams!")
    return streams
