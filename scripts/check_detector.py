import torch
from src.detection.model import TinySSD, DetectorConfig
from src.common.boxes import cxcywh_to_xyxy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = DetectorConfig(input_hw=(320, 320))
model = TinySSD(cfg).to(device)

x = torch.randn(2, 3, 320, 320, device=device)
cls_logits, deltas = model(x)

anchors_cxcywh = model.generate_anchors(device=device)
anchors_xyxy = cxcywh_to_xyxy(anchors_cxcywh)

print("cls_logits:", cls_logits.shape)     # (B, A, 2)
print("deltas:", deltas.shape)            # (B, A, 4)
print("anchors:", anchors_cxcywh.shape)   # (A, 4)
print("anchors_xyxy:", anchors_xyxy.shape)

assert cls_logits.shape[1] == anchors_cxcywh.shape[0]
assert deltas.shape[1] == anchors_cxcywh.shape[0]
print("OK ✅")