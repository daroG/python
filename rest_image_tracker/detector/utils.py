import glob

from cv2 import FONT_HERSHEY_DUPLEX, getTextSize, HOGDescriptor


HOG = HOGDescriptor()

BOX_COLOR = (0, 255, 2)
DETAILS_COLOR = (0, 0, 0)
FONT = FONT_HERSHEY_DUPLEX
FONT_COLOR = (0, 0, 0)
FONT_COLOR_SECONDARY = (255, 255, 255)
FONT_SCALE = 0.5
FONT_THICKNESS = 1


def get_text_size(text: str, font_scale: float, font_thickness: int) -> int:
    """
    Return text font size

    :param text: text to display
    :param font_scale: scale for the text
    :param font_thickness: thickness for the text
    :return: value of the font size
    """
    size, _ = getTextSize(text, FONT, font_scale, font_thickness)
    return size


photos_list = glob.glob('images/*.jpg')
