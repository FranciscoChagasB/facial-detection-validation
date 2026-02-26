import cv2
import torch
from src.detection.model import TinySSD, DetectorConfig
from src.detection.infer import detect_faces

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TinySSD(DetectorConfig(input_hw=(320,320))).to(device)

frame = cv2.imread("teste.jpg")  # BGR
dets, debug = detect_faces(model, frame, device, score_thr=0.6, iou_thr=0.4, assume_bgr=True, return_debug=True)

for d in dets:
    print(d)