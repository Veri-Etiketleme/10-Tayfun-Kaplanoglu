# Third-party modules
import cv2
# Own modules
from background_models.background_model import Model


class WeightedAverageModel(Model):
    __background_model_name__ = "weighted_average"

    def __init__(self, frame, save_folder):
        background_model_frame = frame.copy().astype("float")
        super().__init__(background_model_frame,save_folder, True)

    def get_frame_delta(self, frame):
        # Accumulate the weighted average between the current frame and previous
        # frames, then compute the difference between the current frame and
        # running average
        cv2.accumulateWeighted(frame, self.background_model_frame, 0.5)
        self._save_background_image()
        # TODO: why cv2.convertScaleAbs()?
        return cv2.absdiff(
            frame, cv2.convertScaleAbs(self.background_model_frame))
