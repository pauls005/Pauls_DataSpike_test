import os
import cv2
import torch
import torch.nn.functional as F
import time
from model import DiT_FN
from preprocess import to_tensor


def load_model(weights: str | os.PathLike = "best_model/model_best.pth") -> DiT_FN:
    # 1. Создаём «пустую» модель с нужным числом классов
    model = DiT_FN(
        n_classes=len(class_names), 
        backbone_name="microsoft/dit-base"  # предобученный DiT-backbone
        )

    # 2. Загружаем веса из файла (на тот же девайс, что и модель)
    checkpoint = torch.load(weights, map_location=DEVICE)
    model.load_state_dict(checkpoint)

    # 3. Переносим на DEVICE и переводим в режим инференса
    model.to(DEVICE).eval()
    
    return model


@torch.no_grad()
def predict(model, img_path):
    """
    Инференс: возвращает (idx, name).
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {img_path}")
    
    # BGR->RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = to_tensor(img).unsqueeze(0).to(DEVICE)

    # Получаем эмбеддинг
    emb = model(tensor)
    
    # Если вдруг модель вернула (logits, emb)
    if isinstance(emb, tuple):
        emb = emb[1]
        
    start = time.perf_counter()
    
    # Косинусная классификация без margin
    emb_norm = F.normalize(emb, dim=1)
    weight_norm = F.normalize(model.arcface.weight, dim=1)
    logits = F.linear(emb_norm, weight_norm) * model.arcface.s

    idx = logits.argmax(dim=1).item()
    latency_ms = (time.perf_counter() - start) * 1e3
    name = class_names[idx] if class_names else str(idx)
    return idx, name, latency_ms


if __name__ == "__main__":
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open("classes.txt", 'r') as f:
        class_names = f.read().split()
    
    model = load_model()
    print("Введите путь к изображению паспорта (или 'exit'):")
    
    while True:
        path = input(" > ").strip()
        if path.lower() in ("exit", "quit", "q"):
            break
        if not os.path.isfile(path):
            print("Файл не найден.")
            continue
        try:
            idx, name, t_ms = predict(model, path)
            print(f"Предсказанный класс: {name} | Время инференса: {t_ms:.1f} ms")
        except Exception as e:
            print(f"Ошибка при инференсе: {e}")
