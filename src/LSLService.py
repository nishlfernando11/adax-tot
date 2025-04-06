from pylsl import StreamInlet, resolve_byprop, proc_ALL
import time
import pylsl
import json

class LSLService:
    """
    This class is responsible for managing the LSL (Lab Streaming Layer) service.
    It handles the creation and management of LSL streams for data acquisition.
    """

    def __init__(self, streams):
        """
        Initializes the LSLService instance.
        """
        self.streams = streams
        self.inlet = None
        self.info = None

    def create_stream(self, stream_info):
        """
        Creates a new LSL stream with the given stream information.

        Args:
            stream_info: The information about the stream to be created.

        Returns:
            The created LSL stream.
        """
        pass
    def resolve_streams(self):
        self.streams = pylsl.resolve_streams()
        print(f"Found {len(self.streams)} streams!")

    def get_streams(self):
        if self.streams:
            for s in self.streams:
                print(f"Stream Name: {s.name()}")
                
            inlet = pylsl.StreamInlet(self.streams[0])
            info = inlet.info()
            self.inlet = inlet
            self.info = info
            print("\n--- Stream Metadata ---")
            print(f"Name: {info.name()}")
            print(f"Type: {info.type()}")
            print(f"Channel Count: {info.channel_count()}")
            print(f"Sampling Rate: {info.nominal_srate()}")
            print(f"Channel Format: {info.channel_format()}")
            print("\nListening for data...")
            
            
            