import glob

from cv2 import HOGDescriptor, FONT_HERSHEY_DUPLEX, getTextSize

HOG = HOGDescriptor()

BOX_COLOR = (0, 255, 2)
DETAILS_COLOR = (0, 0, 0)
FONT = FONT_HERSHEY_DUPLEX
FONT_COLOR = (0, 0, 0)
FONT_COLOR_SECONDARY = (255, 255, 255)
FONT_SCALE = 0.5
FONT_THICKNESS = 1


def get_text_size(text, font_scale, font_thickness):
    size, _ = getTextSize(text, FONT, font_scale, font_thickness)
    return size


photos_list = glob.glob('images/*.jpg')
