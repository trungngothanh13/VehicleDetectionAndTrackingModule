import cv2
import numpy as np
from sinks.base_sink import BaseSink
from core.events import FrameTelemetry, ViolationEvent


class PreviewSink(BaseSink):
    """
    Renders live processed frames in a desktop OpenCV window.
    Press 'q' in the window to stop playback.
    """

    def __init__(self, window_name: str = "Vehicle Tracking & Violation Detection"):
        self.window_name = window_name
        self.should_stop = False

    def on_frame(self, frame: np.ndarray, telemetry: FrameTelemetry) -> None:
        cv2.imshow(self.window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[PreviewSink] Live preview interrupted by user.")
            self.should_stop = True

    def on_complete(self) -> None:
        cv2.destroyAllWindows()
