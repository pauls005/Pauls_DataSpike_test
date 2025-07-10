import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import cosine_similarity
from sklearn.metrics import auc
from sklearn.manifold import TSNE
import seaborn as sns
from model import DiT_FN
from preprocess import train_loader, test_loader


def load_model(weights: str | os.PathLike = "best_model/model_best.pth", n_classes=None) -> DiT_FN:
    # 1. Создаём «пустую» модель с нужным числом классов
    model = DiT_FN(
        n_classes=n_classes,
        backbone_name="microsoft/dit-base"
    )
    
    # 2. Загружаем веса из файла (на тот же девайс, что и модель)
    ckpt = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(ckpt)

    # 3. Переносим на DEVICE и переводим в режим инференса
    return model.to(DEVICE).eval()


def compute_embeddings(loader):
    """
    Прогнать весь loader через модель и собрать:

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        Итератор, отдающий батчи (images, labels).  'images' уже должны
        быть приведены к 3×224×224 и нормированы, 'labels' – int-метки.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        embs – 2-D тензор (N, D) с L2-нормированными эмбеддингами
               всех изображений, **на CPU**;
        lbls – 1-D тензор (N,) с соответствующими метками классов,
               также на CPU.
    """
    
    embs, lbls = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            # если модель возвращает tuple (logits, emb)
            if isinstance(out, tuple):
                emb = out[1]
            else:
                emb = out
            embs.append(emb.cpu())
            lbls.append(labels)
    return torch.cat(embs), torch.cat(lbls)


def tsne(embeddings, labels):
    """
    Делает t-SNE на embeddings и возвращает точки.
    embeddings: Tensor [N, D]
    labels:     Tensor [N]
    Возвращает: numpy array shape [N, 2]
    """

    n_samples = embeddings.size(0)
    if n_samples < 5:
        print("Not enough samples for t-SNE.")
        return None

    perp = min(30, max(3, n_samples // 2))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
    points2d = tsne.fit_transform(embeddings.cpu().numpy())
    return points2d


def split_sims(embs, labels):
    """Positive и negative пары внутри одного набора."""
    pos, neg = [], []
    N = embs.size(0)
    for i in range(N):
        for j in range(i+1, N):
            sim = cosine_similarity(embs[i].unsqueeze(0),
                                    embs[j].unsqueeze(0), dim=1).item()
            if labels[i] == labels[j]:
                pos.append(sim)
            else:
                neg.append(sim)
    return pos, neg

def distractor_sims(train_embs, train_lbls, test_embs, test_lbls):
    """Пары train vs test для разных классов.
    В более правильной реализации, здесь должны стоять 
    страны, которых не было в обучении.
    """
    sims = []
    for i in range(train_embs.size(0)):
        for j in range(test_embs.size(0)):
            if train_lbls[i] != test_lbls[j]:
                sim = cosine_similarity(
                    train_embs[i].unsqueeze(0),
                    test_embs[j].unsqueeze(0),
                    dim=1
                ).item()
                sims.append(sim)
    return sims

def compute_ir(pos_s, neg_s, dist_s, fpr):
    """Возвращает (threshold, TPR) при заданном FPR."""
    false_pairs = torch.tensor(neg_s + dist_s)
    sorted_fp, _ = torch.sort(false_pairs, descending=True)
    idx = int(round(fpr * len(sorted_fp)))
    threshold = sorted_fp[idx].item()
    tpr = sum(1 for x in pos_s if x > threshold) / len(pos_s)
    return threshold, tpr

if __name__ == "__main__":

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("classes.txt", 'r') as f:
        class_names = f.read().split()

    # 1) Load model
    n_classes = len(class_names)
    model = load_model(n_classes=n_classes)

    # 2) Compute embeddings + labels
    train_embs, train_lbls = compute_embeddings(train_loader)
    test_embs,  test_lbls  = compute_embeddings(test_loader)

    # 3) Split sims
    pos_sims, neg_sims = split_sims(train_embs, train_lbls)
    dist_sims = distractor_sims(train_embs, train_lbls, test_embs, test_lbls)

    # 4) Compute IR
    fpr_list = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.025, 0.01]
    results = {}
    tpr_vals, thr_vals = [], []
    for fpr in fpr_list:
        thr, tpr = compute_ir(pos_sims, neg_sims, dist_sims, fpr)
        results[f"FPR_{fpr}"] = {"threshold": thr, "TPR": tpr}
        thr_vals.append(thr)
        tpr_vals.append(tpr)

    # 5) Save metrics to JSON
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/tpr_fpr_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # 6) Построение ROC-кривой
    n = 90
    tpr_values = []
    fpr_values = []
    for i in range(n):
        fpr = round(0.001 + i * (1 - 0.001) / n, 3)
        thr, tpr = compute_ir(pos_sims, neg_sims, dist_sims, fpr)
        tpr_values.append(tpr)
        fpr_values.append(fpr)
    fpr_values = np.concatenate([[0], fpr_values, [1]])
    tpr_values = np.concatenate([[0], tpr_values, [1]])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_values, tpr_values, marker='o', label='ROC Curve', color='blue')
    
    # Линия "случайного угадывания"
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
    
    # Закрашиваем область под кривой
    plt.fill_between(fpr_values, tpr_values, alpha=0.2, color='blue', label='AUC Area')
    
    # Настройки осей
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(f'ROC-AUC {round(auc(fpr_values, tpr_values), 3)}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("metrics/roc_auc.png", dpi=300)
    plt.show()

    # 7) heatmap
    
    D = train_embs.size(1)
    # Усредняем эмбеддинги по классам
    class_emb = torch.zeros(n_classes, D)
    for c in range(n_classes):
        mask = (train_lbls == c)
        if mask.any():
            class_emb[c] = train_embs[mask].mean(0)

    # # Нормируем и считаем попарные косинусы
    ce = torch.nn.functional.normalize(class_emb, dim=1)    # [C, D]
    sim_matrix = (ce @ ce.t()).numpy()                      # [C, C]

    # Рисуем и сохраняем heatmap
    plt.figure(figsize=(12,10))
    sns.heatmap(sim_matrix, cmap="viridis", vmin=0, vmax=1)
    plt.title("Class–Class Cosine Similarity")
    plt.xlabel("Class index")
    plt.ylabel("Class index")
    plt.tight_layout()
    plt.savefig("metrics/class_similarity_heatmap.png", dpi=300)
    plt.show()

    # 8) TSNE
    # Compute 2D points
    embeddings_2d = tsne(train_embs, train_lbls)
    if embeddings_2d is not None:
        plt.figure(figsize=(8, 6))
        # Create scatter with class-based colors
        labels_np = train_lbls.cpu().numpy()
        sc = plt.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=labels_np, cmap='viridis', alpha=0.7
        )
        # Add colorbar with class names
        cbar = plt.colorbar(sc, ticks=np.arange(n_classes))
        cbar.ax.set_yticklabels(class_names)
        plt.title("t-SNE of Train Embeddings")
        plt.tight_layout()
        plt.savefig("metrics/TSNE_train.png", dpi=300)
        plt.show()

    embeddings_2d = tsne(test_embs, test_lbls)
    if embeddings_2d is not None:
        plt.figure(figsize=(8, 6))
        # Create scatter with class-based colors
        labels_np = test_lbls.cpu().numpy()
        sc = plt.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=labels_np, cmap='viridis', alpha=0.7
        )
        # Add colorbar with class names
        cbar = plt.colorbar(sc, ticks=np.arange(n_classes))
        cbar.ax.set_yticklabels(class_names)
        plt.title("t-SNE of Test Embeddings")
        plt.tight_layout()
        plt.savefig("metrics/TSNE_test.png", dpi=300)
        plt.show()