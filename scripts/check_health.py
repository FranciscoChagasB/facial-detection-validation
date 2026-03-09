import torch
import torch.nn.functional as F
from src.verification.model import FaceEmbedNet, EmbedConfig

def test_model_collapse(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configurações do novo modelo (base_c=64, emb_dim=512, act="silu")
    cfg = EmbedConfig(base_c=96, emb_dim=512, act="silu", use_se=True)
    model = FaceEmbedNet(cfg).to(device)

    # Carrega o checkpoint
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    print(f"\nCheckpoint carregado: {ckpt_path}")

    # Cria duas 'imagens' de ruído aleatório puro
    img1 = torch.randn(1, 3, 112, 112).to(device)
    img2 = torch.randn(1, 3, 112, 112).to(device)

    with torch.no_grad():
        emb1 = model(img1)
        emb2 = model(img2)

    # Calcula a similaridade (dot product já que estão normalizados L2)
    sim = F.linear(emb1, emb2).item()

    print(f"Similaridade entre os dois ruídos: {sim:.4f}")
    if sim > 0.95:
        print("❌ ALERTA: O modelo sofreu colapso (vetores idênticos).")
    else:
        print("✅ SUCESSO: O modelo está saudável e gerando embeddings dinâmicos!")
        print(f"-> Amostra Vetor 1: {emb1[0][:4].cpu().numpy()}")
        print(f"-> Amostra Vetor 2: {emb2[0][:4].cpu().numpy()}\n")

if __name__ == "__main__":
    # Aponta para o checkpoint que está sendo treinado agora
    test_model_collapse("runs/verification_ms1m_arcface/last.pt")