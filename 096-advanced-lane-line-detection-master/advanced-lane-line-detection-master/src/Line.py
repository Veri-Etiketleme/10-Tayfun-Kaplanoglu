"""Define a class to receive the characteristics of each line detection."""
import numpy as np


class Line():
    """Represents a road lane line."""

    def __init__(self, frame_memory=1, x=None, y=None):
        """Construct the object."""
        # Number of frames to keep
        self.frame_memory = frame_memory
        # was the line detected in the last iteration?
        self.detected = False
        # polynomial averaged over the last n iterations
        self.best_fit = None
        # polynomial for the most recent fit
        self.current_fit = None
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        # Update lane information with given values
        if x is not None and y is not None:
            self.update(x, y)

    def reset(self):
        """Reset line to its initial state"""
        self.detected = False
        self.best_fit = None
        self.current_fit = None
        self.allx = None
        self.ally = None

    def update(self, x, y):
        """Update line information"""
        assert len(x) == len(y)

        self.allx = x
        self.ally = y

        self.current_fit = np.poly1d(np.polyfit(self.allx, self.ally, 2))

        if self.best_fit is None:
            self.best_fit = self.current_fit
        else:
            # Average polynomial coefficients
            current_coeffs = self.current_fit.c
            best_coeffs = self.best_fit.c
            self.best_fit = np.poly1d((best_coeffs * (self.frame_memory - 1) + current_coeffs) / self.frame_memory)
