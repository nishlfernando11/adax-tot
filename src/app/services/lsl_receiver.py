from pylsl import StreamInlet, resolve_byprop, proc_ALL
import time
import pylsl
import json

streams = pylsl.resolve_streams()
print(f"Found {len(streams)} streams!")

if streams:
    for s in streams:
        print(f"Stream Name: {s.name()}")
        
    inlet = pylsl.StreamInlet(streams[0])
    info = inlet.info()
    print("\n--- Stream Metadata ---")
    print(f"Name: {info.name()}")
    print(f"Type: {info.type()}")
    print(f"Channel Count: {info.channel_count()}")
    print(f"Sampling Rate: {info.nominal_srate()}")
    print(f"Channel Format: {info.channel_format()}")
    print("\nListening for data...")
    while True:
        sample, timestamp = inlet.pull_sample()
        print(f"Received: {sample} at {timestamp:.6f}")