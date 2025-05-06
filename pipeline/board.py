import cv2
import numpy as np


def open_drawing_board(save_path="board_output.png", width=800, height=200):
    """
    Opens a simple drawing board for freehand text input using OpenCV.

    Allows the user to draw with the mouse and save the result as an image.

    Controls:
        - Left Mouse Button: Press and drag to draw
        - 's' key: Save the drawing and close the window
        - 'Esc' key: Exit without saving

    Args:
        save_path (str): File path where the drawn image will be saved.
        width (int): Width of the drawing canvas in pixels.
        height (int): Height of the drawing canvas in pixels.
    """

    drawing = False
    canvas = np.ones((height, width), dtype=np.uint8) * 255
    last_point = None

    def draw(event, x, y, flags, param):
        nonlocal drawing, last_point
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cv2.line(canvas, last_point, (x, y), color=0, thickness=4)
            last_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            last_point = None

    cv2.namedWindow("Draw Text", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Draw Text", draw)

    while True:
        cv2.imshow("Draw Text", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to cancel
            break
        elif key == ord("s"):  # press 's' to save and exit
            cv2.imwrite(save_path, canvas)
            break

    cv2.destroyAllWindows()
