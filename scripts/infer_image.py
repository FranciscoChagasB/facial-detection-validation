import cv2
import torch

from src.detection.model import TinySSD, DetectorConfig
from src.detection.infer import detect_faces

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TinySSD(DetectorConfig(input_hw=(320,320))).to(device)

ckpt = torch.load("./runs/detector_smoke/last.pt", map_location="cpu")
model.load_state_dict(ckpt["model"], strict=True)

img_path = "scripts\png-transparent-crowd-businessperson-generativity-others-company-people-social-group.png"
frame = cv2.imread(str(img_path))  # BGR
if frame is None:
    raise FileNotFoundError(f"Imagem não encontrada: {img_path}")

dets, debug = detect_faces(
    model=model,
    frame=frame,
    device=device,
    score_thr=0.25,
    iou_thr=0.60,
    assume_bgr=True,
    return_debug=True
)

print("debug:", debug)
print("num dets:", len(dets))

# desenha e salva
out = frame.copy()
for d in dets:
    x1,y1,x2,y2 = map(int, [d.x1, d.y1, d.x2, d.y2])
    cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(out, f"{d.score:.2f}", (x1, max(0,y1-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

cv2.imwrite("out_det.jpg", out)
print("salvo: out_det.jpg")