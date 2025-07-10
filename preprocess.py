import albumentations as A
import cv2
from typing import Any
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

NORM = A.Normalize(mean=(0.485,0.456,0.406),  # нормализация по каналам
                   std =(0.229,0.224,0.225))


def letterbox(size=(224,224), border_val=0):
    """
    Resize-&-pad трансформация с сохранением пропорций.

    Parameters
    ----------
    size : Tuple[int, int], default (224, 224)
        Итоговые (width, height) картинки после преобразования.
    border_val : int, default 0
        Цвет заливки для свободных полей. 0 — чёрный.
    Returns
    -------
    albumentations.Compose
        Пайплайн:
        1. LongestMaxSize — сжатие/растяжение, чтобы длинная
           сторона стала равна max(size).
        2. PadIfNeeded  — добавление рамок до точного размера.
    """
    w, h = size
    return A.Compose([
        A.LongestMaxSize(max_size=max(w, h)),
        A.PadIfNeeded(min_height=h, min_width=w,
                      border_mode=cv2.BORDER_CONSTANT,
                      value=border_val)
    ])


def to_tensor(image):
    """
    Приводит RGB-изображение к формату, пригодному для подачи в модель.

    Пайплайн:
        1. letterbox((224, 224)) – вписать изображение в 224×224
           без изменения пропорций, недостающие поля — чёрные.
        2. NORM – нормализация по каналам (mean/std от ImageNet).
        3. ToTensorV2() – HWC -> CHW, float32, torch.Tensor.

    Parameters
    ----------
    image : np.ndarray
        Цветное RGB-изображение.

    Returns
    -------
    torch.Tensor
        Тензор формы (3, 224, 224), dtype torch.float32,
        нормированный и готовый для инференса/обучения.
    """
    tf = A.Compose([letterbox((224,224)), NORM, ToTensorV2()])
    return tf(image=image)["image"]


class PassportDataset(Dataset):
    """
    Папка-ориентированный датасет изображений документов.
    Parameters
    ----------
    root : str | Path
        Корневая папка: каждый под-каталог = отдельная страна (класс).
    train : bool, default True
        True  -> использовать train_transform;
        False -> val_transform.
    """

    
    def __init__(self, root: str, train: bool = True):
        self.train = train
        self.root = Path(root)
        # собираем пути и метки
        self.classes = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        self.cls2idx = {c:i for i,c in enumerate(self.classes)}
        self.files, self.labels = [], []
        for c in self.classes:
            for img_path in (self.root/c).glob("*.jpg"):
                self.files.append(str(img_path))
                self.labels.append(self.cls2idx[c])
        # выбираем трансформ
        self.tf = train_transform if self.train else val_transform

    
    def __len__(self):
        return len(self.files)

    
    def __getitem__(self, idx):
        img = cv2.imread(self.files[idx])[:,:,::-1]  # BGR->RGB
        # img = strip_mrz(img)
        img = self.tf(image=img)["image"]           # np->Tensor, C×H×W
        label = self.labels[idx]
        return img, label

## strong augmentation, uncomment if it's needed
# train_transform = A.Compose([
    # A.Rotate(limit=10, border_mode=cv2.BORDER_REPLICATE, p=0.5),
    # A.RandomRotate90(p=0.2),
    # A.Perspective(scale=(0.05,0.15), p=0.3),
    # A.RandomBrightnessContrast(0.2,0.2, p=0.4),
    # A.MotionBlur(p=0.2),
    # A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),           # добавили шум
    # A.ImageCompression(quality_lower=80, quality_upper=100, p=0.3),  # артефакты JPEG
    # letterbox((224,224)),
    # NORM,
    # ToTensorV2(),
# ])

## light augmentation
train_transform = A.Compose([
    # повороты стали мягче: limit 7deg вместо 10deg, p=0.3 вместо 0.5
    A.Rotate(limit=7, border_mode=cv2.BORDER_REPLICATE, p=0.35),
    # реже меняем ориентацию на 90deg
    A.RandomRotate90(p=0.14),  # 0.2 -> 0.14 (30% вниз)
    # уменьшаем искажение перспективы
    A.Perspective(scale=(0.03, 0.10), p=0.21),  # (0.05,0.15)->(0.03,0.10), p=0.3->0.21
    # ослабляем контраст и яркость
    A.RandomBrightnessContrast(brightness_limit=0.16,
                               contrast_limit=0.16,
                               p=0.3),          # 0.4->0.3
    # реже размытие движения
    A.MotionBlur(p=0.1),       # 0.2 -> 0.1
    # шума чуть меньше
    A.GaussNoise(var_limit=(3.0, 20.0), p=0.2),  # (5,30)->(3,20), p=0.3->0.21
    # JPEG-артефакты в меньшей силе
    A.ImageCompression(quality_lower=85,
                       quality_upper=100,
                       p=0.2),          # 0.3->0.21
    letterbox((224,224)),
    NORM,
    ToTensorV2(),
])

## strong augmentations
val_transform = A.Compose([
    letterbox((224,224)),
    A.RandomBrightnessContrast(0.05,0.05, p=0.2),
    A.Rotate(limit=5, border_mode=cv2.BORDER_REPLICATE, p=0.2),
    NORM,
    ToTensorV2(),
])

## light (no) augmentations
# val_transform = A.Compose([
#     letterbox((224,224)),
#     NORM,
#     ToTensorV2(),
# ])


# 1) создаём полный датасет (train=True, чтобы он использовал AUG_TRAIN)
path = 'dataset/dataset'
full_ds = PassportDataset(path, train=True)

# 2) берём все индексы и метки
indices = list(range(len(full_ds)))
labels  = full_ds.labels

# 3) сначала вычленяем train (80%) и temp (20%)
train_idx, temp_idx, _, temp_lbls = train_test_split(
    indices, labels,
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# 4) из temp делим поровну на val и test (16% val и 4% test)
val_idx, test_idx, _, _ = train_test_split(
    temp_idx, temp_lbls,
    test_size=0.2,
    stratify=temp_lbls,
    random_state=42
)

# 5) subsets с разными флагами train/val
train_ds = Subset(PassportDataset(path, train=True),  train_idx)
val_ds   = Subset(PassportDataset(path, train=False), val_idx)
test_ds  = Subset(PassportDataset(path, train=False), test_idx)

# 6) DataLoaders
batch_size = 32
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

# 7) Создаем файл с названиями классов
with open("classes.txt", "w", encoding="utf-8") as f:
    for c in train_loader.dataset.dataset.classes:
        f.write(f'{c}\n')