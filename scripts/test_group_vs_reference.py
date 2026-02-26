import os
import cv2
import numpy as np
import torch

from src.detection.model import TinySSD, DetectorConfig
from src.detection.infer import detect_faces
from src.verification.model import EmbedConfig
from src.verification.infer import load_verification_model, extract_embedding, preprocess_face_image
from src.common.preprocess import crop_faces
from src.common.preprocess import expand_and_square_boxes_xyxy_px

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Ajuste paths ----------
    group_path = r"scripts\png-transparent-crowd-businessperson-generativity-others-company-people-social-group.png"
    reference_path = r"scripts\shutterstock_117552124_Web.jpg"

    det_ckpt = r"runs\detector_smoke\last.pt"                 # troque pelo seu detector treinado
    ver_ckpt = r"runs\verification_ms1m_v2\last.pt"           # seu embedding treinado

    out_img_path = r"out_group_result.jpg"

    # ---------- Configs ----------
    score_thr = 0.35
    iou_thr = 0.50
    threshold_cosine = 0.60   # ajuste depois

    # Detector (tem que bater com o treino do detector)
    det_model = TinySSD(DetectorConfig(input_hw=(320, 320))).to(device)
    if os.path.isfile(det_ckpt):
        st = torch.load(det_ckpt, map_location="cpu")
        det_model.load_state_dict(st["model"] if "model" in st else st, strict=True)
    det_model.eval()

    # Verification (tem que bater com o treino v2: base_c=96 emb_dim=512)
    ver_model = load_verification_model(
        checkpoint_path=ver_ckpt,
        device=device,
        cfg=EmbedConfig(input_hw=(112,112), base_c=64, emb_dim=256, act="relu", use_se=True, dropout=0.0),
        strict=True
    )
    ver_model.eval()

    # ---------- Carrega imagens ----------
    frame_bgr = cv2.imread(group_path)
    if frame_bgr is None:
        raise FileNotFoundError(f"Não consegui ler: {group_path}")

    ref_bgr = cv2.imread(reference_path)
    if ref_bgr is None:
        raise FileNotFoundError(f"Não consegui ler: {reference_path}")

    # Converte para RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)

    # ---------- Detecta faces no grupo ----------
    dets, debug = detect_faces(
        model=det_model,
        frame=frame_bgr,
        device=device,
        score_thr=score_thr,
        iou_thr=iou_thr,
        assume_bgr=True,
        return_debug=True
    )

    print("Debug:", debug)
    print("Faces detectadas:", len(dets))
    if len(dets) == 0:
        print("Nenhuma face detectada. Tente reduzir score_thr para 0.2")
        return

    H, W = frame_rgb.shape[:2]
    boxes_px = np.array([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=np.float32)
    boxes_px = expand_and_square_boxes_xyxy_px(boxes_px, img_w=W, img_h=H, scale=1.25)
    crops = crop_faces(frame_rgb, boxes_px, out_size=112, min_side=20)
    
    if len(crops) == 0:
        print("Nenhum crop válido. Tente reduzir min_side.")
        return

    # ---------- Embedding da referência ----------
    ref_emb = extract_embedding(
        model=ver_model,
        image=ref_rgb,
        device=device,
        assume_bgr=False,
        center_crop_square=True,   # ajuda se sua referência não for crop perfeito
        return_tensor=True
    ).to(device)

    # ---------- Compara referência com cada face detectada ----------
    best_score = -1.0
    best_idx = -1
    scores = []

    with torch.no_grad():
        # gera embeddings do grupo em batch
        xs = []
        for c in crops:
            x = preprocess_face_image(
                image=c,
                input_hw=(112,112),
                device=device,
                assume_bgr=False,
                center_crop_square=False
            )
            xs.append(x)
        xb = torch.cat(xs, dim=0)          # (N,3,112,112)
        emb_group = ver_model(xb)          # (N,D) normalizado

        # cosine = dot product
        sims = (emb_group @ ref_emb.view(-1,1)).squeeze(1)   # (N,)
        sims_np = sims.detach().cpu().numpy().astype(float).tolist()

    for i, s in enumerate(sims_np):
        scores.append(s)
        if s > best_score:
            best_score = s
            best_idx = i

    print("\nTop-1 match:")
    print("  idx:", best_idx)
    print("  cosine:", best_score)
    print("  aprovado:", best_score >= threshold_cosine, " (thr=", threshold_cosine, ")")

    # ---------- Desenha resultado ----------
    out = frame_bgr.copy()

    # desenha todas as faces + score
    for i, d in enumerate(dets[:len(scores)]):
        x1,y1,x2,y2 = map(int, [d.x1, d.y1, d.x2, d.y2])
        s = scores[i]
        color = (0, 255, 0) if s >= threshold_cosine else (0, 0, 255)
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        cv2.putText(out, f"{s:.2f}", (x1, max(0,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # destaca best
    if 0 <= best_idx < len(dets):
        d = dets[best_idx]
        x1,y1,x2,y2 = map(int, [d.x1, d.y1, d.x2, d.y2])
        cv2.rectangle(out, (x1,y1), (x2,y2), (255, 255, 0), 3)
        cv2.putText(out, "BEST", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.imwrite(out_img_path, out)
    print("\nImagem salva:", out_img_path)

if __name__ == "__main__":
    main()