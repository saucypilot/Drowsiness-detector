import cv2


def window_is_open(name: str) -> bool:
    """Return True if an OpenCV HighGUI window exists and is visible."""
    try:
        prop = cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE)
        return prop >= 1
    except cv2.error:
        return False
