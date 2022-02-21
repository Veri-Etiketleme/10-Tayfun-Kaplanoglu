# Third-party modules
import cv2
# Own modules
from background_models.background_model import Model


class FirstFrameModel(Model):
    __background_model_name__ = "first_frame"

    def __init__(self, background_model_frame, save_folder):
        super().__init__(background_model_frame, save_folder, False)

    def get_frame_delta(self, frame):
        # Compute the absolute difference between the current frame and first
        # frame
        frame_delta = cv2.absdiff(self.background_model_frame, frame)
        self._save_background_image()
        return frame_delta
