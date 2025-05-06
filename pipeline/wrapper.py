import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "SimpleHTR"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "CRAFT-pytorch"))
sys.path.append(
    str(Path(__file__).resolve().parent.parent / "deep-text-recognition-benchmark")
)
sys.path.append(
    str(
        Path(__file__).resolve().parent.parent
        / "CTCWordBeamSearch"
        / "build"
        / "lib.win-amd64-3.8"
    )
)

import cv2
import numpy as np
import torch
import tensorflow as tf
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset

from dataset import AlignCollate
from inference import run_craft_inference, merge_boxes, draw_boxes
from model import Model as ClovaAI
from refinenet import RefineNet
from craft import CRAFT
from src.model import Model as SimpleHTR, DecoderType
from src.dataloader_arm import Batch
from src.preprocessor import Preprocessor
from word_beam_search import WordBeamSearch


from utils import CTCLabelConverter
from PIL import Image


"""
OCR Wrapper Module for Armenian Handwritten Text Recognition

Supports two recognition models:
- SimpleHTR (TensorFlow)
- ClovaAI (PyTorch)

Uses CRAFT for text detection and supports Word Beam Search, Beam Search, and Best Path decoding.
"""


# The Parameter class for ClovaAI model configuration
class Params:
    """
    Configuration class for the ClovaAI model setup.

    Holds all hyperparameters and model options used for model initialization,
    including architecture choices, input sizes, character list, etc.
    """

    def __init__(self):
        self.num_fiducial = 20
        self.input_channel = 1
        self.rgb = False
        self.output_channel = 512
        self.hidden_size = 256
        self.Transformation = "TPS"
        self.FeatureExtraction = "ResNet"
        self.SequenceModeling = "BiLSTM"
        self.Prediction = "CTC"
        self.num_class = 100
        self.batch_max_length = 32
        self.character = "ԱԲԳԴԵԶԷԸԹԺԻԼԽԾԿՀՁՂՃՄՅՆՇՈՉՊՋՌՍՎՏՐՑՒՓՔՕՖաբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքևօֆ՝՝1234567890՝՜:՞«»,.-()"
        self.sensitive = True
        self.imgW = 100
        self.imgH = 32
        self.PAD = True
        self.workers = 0
        self.data_filtering_off = True


# Custom class to help with the ClovaAI inference
class ImageDataset(Dataset):
    """
    Simple PyTorch Dataset wrapper for a list of PIL images.

    Returns dummy labels for compatibility with the model's DataLoader.
    """

    def __init__(self, image_list):
        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        return self.image_list[idx], "dummy"


def copyStateDict(state_dict):
    """
    Strips 'module.' prefix from keys if model was trained with DataParallel.

    Args:
        state_dict (dict): Original model state dict.

    Returns:
        OrderedDict: Cleaned state dict ready for loading.
    """

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


def load_detector(craft_weights_path, use_refiner=False, refiner_weights_path=None):
    """
    Loads the CRAFT text detection model and optionally the refinement network.

    Args:
        craft_weights_path (str): Path to the CRAFT weights.
        use_refiner (bool): Whether to load the refinement model.
        refiner_weights_path (str, optional): Path to the refiner weights.

    Returns:
        Tuple[CRAFT, RefineNet or None]: Loaded detector and optional refiner.
    """

    net = CRAFT()
    state_dict = torch.load(
        craft_weights_path, map_location="cuda" if torch.cuda.is_available() else "cpu"
    )
    net.load_state_dict(copyStateDict(state_dict))

    if torch.cuda.is_available():
        net = net.cuda()
    net.eval()

    refine_net = None
    if use_refiner and refiner_weights_path:
        refine_net = RefineNet()
        refiner_state = torch.load(
            refiner_weights_path,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
        )
        refine_net.load_state_dict(copyStateDict(refiner_state))
        if torch.cuda.is_available():
            refine_net = refine_net.cuda()
        refine_net.eval()

    return net, refine_net


def load_recognizer(
    model_path,
    model_name="SimpleHTR",
    char_list_path=None,
    corpus_path=None,
    decoder=None,
):
    """
    Loads the OCR recognizer model, either SimpleHTR or ClovaAI.

    Args:
        model_path (str): Path to model checkpoint.
        model_name (str): One of ["SimpleHTR", "ClovaAI"].
        char_list_path (str, optional): Path to character list (SimpleHTR only).
        corpus_path (str, optional): Path to corpus file (SimpleHTR only).
        decoder (str, optional): One of ["wbs", "bs", "bestpath"].

    Returns:
        Model: Loaded OCR model.
    """

    if model_name == "SimpleHTR":
        with open(char_list_path, encoding="utf-8") as f:
            char_list = list(f.read())

        if decoder == "wbs":
            decoder_type = DecoderType.WordBeamSearch
        elif decoder == "bs":
            decoder_type = DecoderType.Beamsearch
        else:
            decoder_type = DecoderType.BestPath

        tf.compat.v1.reset_default_graph()
        model = SimpleHTR(
            char_list=char_list,
            must_restore=True,
            fine_tune=False,
            specified_model_path=model_path,
            decoder_type=decoder_type,
            corpus_path=corpus_path,
        )
    elif model_name == "ClovaAI":
        opt = Params()
        model = ClovaAI(opt)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

    return model


def recognize_from_image(image_path, detector, refine_net, recognizer):
    """
    Runs full OCR pipeline: detection with CRAFT and recognition with selected model.

    Args:
        image_path (str): Path to the image for OCR.
        detector (CRAFT): Loaded detection model.
        refine_net (RefineNet or None): Optional refinement network.
        recognizer (Model): OCR model (SimpleHTR or ClovaAI).

    Returns:
        Tuple[str, np.ndarray]: Recognized text string and annotated image with boxes.
    """

    # Run detection
    image, boxes = run_craft_inference(image_path, detector, refine_net)
    boxes = sort_boxes_reading_order(merge_boxes(boxes))

    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cropped_images = []

    for box in boxes:
        x_min = int(min(pt[0] for pt in box))
        x_max = int(max(pt[0] for pt in box))
        y_min = int(min(pt[1] for pt in box))
        y_max = int(max(pt[1] for pt in box))
        cropped = img_gray[y_min:y_max, x_min:x_max]
        if cropped is not None and cropped.size > 0:
            cropped_images.append(Image.fromarray(cropped).convert("L"))

    if not cropped_images:
        return "", draw_boxes(image, boxes)

    if isinstance(recognizer, torch.nn.Module):
        opt = Params()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        converter = CTCLabelConverter(opt.character)
        raw_preds = predict(cropped_images, recognizer, opt, device, converter)
    else:
        raw_preds = []
        preprocessor = Preprocessor(img_size=(128, 32), dynamic_width=True, padding=16)
        for img in cropped_images:
            processed = preprocessor.process_img(np.array(img.convert("L")))
            batch = Batch([processed], None, 1)
            pred, _ = recognizer.infer_batch(batch, calc_probability=False)
            raw_preds.append(pred[0])

    vis_image = draw_boxes(image, boxes)
    return " ".join(raw_preds), vis_image


def sort_boxes_reading_order(boxes, y_thresh=50):
    """
    Sorts detected word boxes into reading order (left-to-right, top-to-bottom).

    Args:
        boxes (list): List of bounding boxes.
        y_thresh (int): Vertical threshold to group boxes into the same line.

    Returns:
        list: Sorted list of bounding boxes.
    """

    def get_midpoint(box):
        x = np.mean([pt[0] for pt in box])
        y = np.mean([pt[1] for pt in box])
        return x, y

    boxes_with_mids = [(box, *get_midpoint(box)) for box in boxes]
    boxes_with_mids.sort(key=lambda x: x[2])

    lines = []
    for box, x, y in boxes_with_mids:
        matched = False
        for line in lines:
            avg_y = np.mean([b[2] for b in line])  # average y of current line
            if abs(y - avg_y) < y_thresh:
                line.append((box, x, y))
                matched = True
                break
        if not matched:
            lines.append([(box, x, y)])

    sorted_boxes = []
    for line in lines:
        line.sort(key=lambda x: x[1])  # sort by x midpoint
        sorted_boxes.extend([b for b, _, _ in line])

    return sorted_boxes


def predict(images, model, opt, device, converter):
    """
    Performs OCR on a list of cropped word images using ClovaAI.

    Args:
        images (list of PIL.Image): List of input images.
        model (torch.nn.Module): Loaded OCR model.
        opt (Params): Configuration parameters.
        device (torch.device): Torch device (cpu or cuda).
        converter (CTCLabelConverter): Label converter for decoding predictions.

    Returns:
        List[str]: Recognized text for each input image.
    """

    image_data = ImageDataset(images)
    collate = AlignCollate(
        imgH=opt.imgH,
        imgW=opt.imgW,
        keep_ratio_with_pad=opt.PAD,
    )
    image_loader = DataLoader(
        dataset=image_data,
        batch_size=len(images),
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=collate,
        pin_memory=(device.type != "cpu"),
    )

    predictions = []

    model.eval()
    with torch.no_grad():
        for image_tensors, _ in image_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)

            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(
                device
            )
            prediction_text = (
                torch.LongTensor(batch_size, opt.batch_max_length + 1)
                .fill_(0)
                .to(device)
            )

            if "CTC" in opt.Prediction:
                prediction = model(input=image, text=prediction_text).log_softmax(2)

                prediction_size = torch.IntTensor([prediction.size(1)] * batch_size)
                _, prediction_indices = prediction.permute(1, 0, 2).max(2)
                prediction_indices = prediction_indices.transpose(1, 0).contiguous()

                prediction_str = converter.decode(
                    prediction_indices.data, prediction_size.data
                )
            else:
                prediction = model(input=image, text=prediction_text, is_train=False)

                _, prediction_indices = prediction.max(2)
                prediction_str = converter.decode(prediction_indices, length_for_pred)

            soft_prob = torch.nn.functional.softmax(prediction, dim=2)
            soft_prob, index = soft_prob.max(2)

            predicted_texts = []
            for idx, predicted_text in enumerate(prediction_str):
                if "Attn" in opt.Prediction:
                    eos_index = predicted_text.find("[s]")
                    box_prob = torch.mean(soft_prob[idx][:eos_index]).item()
                    if box_prob < 0.7:
                        predicted_text = ""
                    else:
                        predicted_text = predicted_text[:eos_index]
                predicted_texts.append(predicted_text)

            predictions += predicted_texts

    return predictions


def get_model_config(model_name="SimpleHTR", decoder_name="wbs"):
    """
    Returns config dictionary based on selected model name and decoder.

    Args:
        model_name (str): One of ["SimpleHTR", "ClovaAI"].
        decoder_name (str): Decoder type for SimpleHTR (wbs, bs, bestpath).

    Returns:
        dict: Configuration dictionary for model loading.
    """

    config_dict = {}
    if model_name == "SimpleHTR":
        config_dict["model_path"] = "../SimpleHTR/model_checkpoints_armo/"
        config_dict["char_list_path"] = (
            "../SimpleHTR/model_checkpoints_armo/charList.txt"
        )
        config_dict["decoder"] = decoder_name
        config_dict["corpus_path"] = "../SimpleHTR/data/corpus.txt"

    elif model_name == "ClovaAI":

        best_model_path = (
            "../deep-text-recognition-benchmark/saved_models/best_model_epoch_27.pth"
        )
        config_dict["model_name"] = model_name
        config_dict["model_path"] = best_model_path
    return config_dict
