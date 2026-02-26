import os
import torch

from src.verification.model import EmbedConfig
from src.verification.infer import (
    load_verification_model,
    verify_faces,
    build_gallery_from_folder,
    extract_embedding,
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = r"runs\verification_ms1m_v2\last.pt"

    # IMPORTANTE: tem que bater com o treino v2: base_c=96 emb_dim=512
    model = load_verification_model(
        checkpoint_path=ckpt_path,
        device=device,
        cfg=EmbedConfig(input_hw=(112, 112), base_c=64, emb_dim=256, act="relu", use_se=True, dropout=0.0),
        strict=True,
    )

    print("Model carregado:", ckpt_path)
    print("Device:", device)

    # -----------------------------
    # A) Teste de pares (verificação)
    # -----------------------------
    # Ajuste os caminhos para seus arquivos de teste:
    same_a = r"scripts\rosto_rg.jpg"
    same_b = r"scripts\rosto_dia.jpg"  # mesma pessoa
    diff_a = r"scripts\shutterstock_117552124_Web.jpg"
    diff_b = r"scripts\shutterstock_117552124_Web.jpg"  # pessoa diferente

    # threshold inicial sugerido (cosine)
    thr = 0.60

    if os.path.isfile(same_a) and os.path.isfile(same_b):
        res_same = verify_faces(
            model=model,
            image1=same_a,
            image2=same_b,
            device=device,
            metric="cosine",
            threshold=thr,
        )
        print("\n[PAIR] MESMA pessoa")
        print("img1:", same_a)
        print("img2:", same_b)
        print("result:", res_same)
    else:
        print("\n[PAIR] Pulando teste MESMA pessoa (arquivos não encontrados).")

    if os.path.isfile(diff_a) and os.path.isfile(diff_b):
        res_diff = verify_faces(
            model=model,
            image1=diff_a,
            image2=diff_b,
            device=device,
            metric="cosine",
            threshold=thr,
        )
        print("\n[PAIR] PESSOAS diferentes")
        print("img1:", diff_a)
        print("img2:", diff_b)
        print("result:", res_diff)
    else:
        print("\n[PAIR] Pulando teste DIFERENTES (arquivos não encontrados).")

    print("\nDica: se SAME estiver dando False, baixe threshold (ex: 0.55).")
    print("      se DIFF estiver dando True, suba threshold (ex: 0.65).")

    # -----------------------------
    # B) Teste com galeria (base)
    # -----------------------------
    # Estrutura esperada:
    # data\galeria\
    #   12345678900\foto.jpg
    #   98765432100\foto.jpg
    """
    gallery_root = r"data\galeria"
    query_path = r"data\test_faces\consulta.jpg"

    if os.path.isdir(gallery_root) and os.path.isfile(query_path):
        gallery = build_gallery_from_folder(
            model=model,
            root_dir=gallery_root,
            device=device,
            metric="cosine",
            one_image_per_identity=True,
        ).to(device)

        query_emb = extract_embedding(model, query_path, device=device)

        topk = gallery.search(query_emb, top_k=5)
        print("\n[GALLERY] Consulta:", query_path)
        print("Top-5:")
        for m in topk:
            print(f"  #{m.rank}  id={m.identity}  score={m.score:.4f}")

        top1, ok = gallery.verify_top1(query_emb, threshold=thr)
        print("\nTop-1:", top1)
        print("MATCH (thr=%.2f): %s" % (thr, ok))
    else:
        print("\n[GALLERY] Pulando (faltou pasta data\\galeria ou arquivo consulta.jpg).")
        print("Crie a estrutura:")
        print(r"  data\galeria\<CPF>\foto.jpg")
        print(r"e um arquivo query: data\test_faces\consulta.jpg")
    """

if __name__ == "__main__":
    main()