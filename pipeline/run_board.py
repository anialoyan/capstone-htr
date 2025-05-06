from board import open_drawing_board
from wrapper import load_detector, load_recognizer, recognize_from_image, get_model_config
import os
import time


"""
Terminal-Based Handwritten OCR Inference Tool

This script launches a freehand drawing board for the user to write Armenian text by hand.
It supports two OCR models (SimpleHTR and ClovaAI) for recognizing the drawn content.

Features:
- CRAFT-based word detection with optional refiner
- Choice between SimpleHTR (TensorFlow) and ClovaAI (PyTorch) recognizers
- Interactive loop: draw → recognize → repeat
- Saves the board drawing to an image and runs end-to-end OCR on it

Controls:
- Draw with left mouse click + drag
- Press 's' to save and exit the drawing window
- Press 'Esc' to exit without saving

Usage:
    python run_board.py
"""

save_path = './board_output.png'

detector, refine_net = load_detector(
    craft_weights_path='../CRAFT-pytorch/weights/craft_mlt_25k.pth',
    use_refiner=True,
    refiner_weights_path='../CRAFT-pytorch/weights/craft_refiner_CTW1500.pth'
)

print("Select OCR model:")
print("1. SimpleHTR")
print("2. ClovaAI")

choice = input("Enter 1 or 2: ").strip()
if choice == "1":
    model_name = "SimpleHTR"
    decoder = input("Decoder (wbs / bestpath / beamsearch) [default: wbs]: ").strip() or "wbs"
    config = get_model_config(model_name, decoder)
elif choice == "2":
    model_name = "ClovaAI"
    config = get_model_config(model_name)
else:
    print("Invalid selection. Exiting.")
    exit()

recognizer = load_recognizer(**config)

while True:
    print("\nDraw your text on the board. Close the window when done.")
    open_drawing_board(save_path=save_path)

    if not os.path.exists(save_path):
        print("No image saved. Try again.")
        continue

    time.sleep(0.3)  # ensure file is saved

    result, _ = recognize_from_image(save_path, detector, refine_net, recognizer)
    print("\nRecognized Text:\n", result)

    user_input = input("\nWant to try again? (y/n): ").strip().lower()
    if user_input == 'n':
        break

