import os
# Own modules
from utilities.utils import write_image


# Abstract background model
class Model:
    def __init__(self, background_model_frame, saving_cfg=None,
                 update_background_image=False):
        self.background_model_frame = background_model_frame
        self.saving_cfg = saving_cfg
        self.update_background_image = update_background_image
        self.saving = True
        self.count_save = 0

    def get_frame_delta(self, frame):
        raise NotImplementedError

    def _save_background_image(self):
        # Save background image
        if self.saving_cfg.get('saved_folder') and self.saving:
            inum = "{0:06d}".format(self.count_save)
            bi_fname = "background_image_{}.{}".format(
               inum, self.saving_cfg.get('image_format', 'png'))
            self.count_save += 1
            bi_fname = os.path.join(self.saving_cfg.get('saved_folder'), bi_fname)
            write_image(bi_fname, self.background_model_frame)
            if not self.update_background_image:
                self.saving = False
