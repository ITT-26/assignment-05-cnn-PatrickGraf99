from enum import Enum


class CameraAction(Enum):
    APPLY_GRAYSCALE = 0,
    APPLY_PORTRAIT = 1,
    APPLY_CANNY = 2,
    ACTIVATE_SELFIE_TIMER = 3