import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import pi
from transformers import AutoConfig, AutoModel

"""
DiT_FN = DiT-base + ArcFace

Backbone: DiT-base (примерно 86 M параметров, patch = 16). Attention схватывает разную верстку документов; с gradient-checkpointing и частичным fine-tune укладывается даже в 4 ГБ GPU — доучиваем лишь верхние блоки.

Пулинг: среднее по patch-токенам → линейная проекция 512 d → L2-норма.

Head: ArcFace (s = 30, m = 0.5) даёт плотные внутриклассовые кластеры и устойчивую шкалу логитов на сотни стран.

Обучение — x, y → logits, emb; инференс — x → emb
"""


class DiT_FN(nn.Module):
    """
    Fine-tuning model: DiT backbone + ArcFace head.
    """
    def __init__(
        self,
        n_classes: int = 24,
        backbone_name: str = "microsoft/dit-base",
        embed_dim: int = 512,
        s: float = 30.0,
        m: float = 0.5,
        drop_p: float = 0.3
    ):
        super().__init__()
        # Dropout для эмбеддингов
        self.dropout = nn.Dropout(drop_p)

        # Загружаем предобученный DiT-бэкбон
        config = AutoConfig.from_pretrained(backbone_name)
        self.backbone = AutoModel.from_pretrained(backbone_name, config=config)

        # Глобальный пуллинг по токенам (без [CLS])
        hidden_size = self.backbone.config.hidden_size
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Проекция в эмбеддинги
        self.embedding = nn.Linear(hidden_size, embed_dim, bias=False)

        # ArcFace-голова для классификации
        self.arcface = ArcFaceLoss(
            in_features=embed_dim,
            out_features=n_classes,
            s=s,
            m=m
        )

        # Включаем gradient checkpointing для экономии памяти (опционально)
        try:
            self.backbone.gradient_checkpointing_enable()
        except AttributeError:
            pass

    def freeze_backbone(self):
        """Заморозить все параметры бэкбона."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, n_last_layers: int = 2):
        """Разморозить последние n_last_layers блоков энкодера."""
        # Определяем, где лежат слои энкодера
        if hasattr(self.backbone, 'encoder'):
            layers = self.backbone.encoder.layer
        else:
            layers = self.backbone.vision_model.encoder.layer
        # Включаем requires_grad только для последних слоев
        for i, block in enumerate(layers):
            requires = i >= len(layers) - n_last_layers
            for param in block.parameters():
                param.requires_grad = requires

    def forward(self, pixel_values: torch.Tensor, labels: torch.Tensor = None):
        """
        Args:
            pixel_values: Tensor[B, C, H, W] — входные изображения
            labels:       Tensor[B] — индексы классов (для обучения)
        Returns:
            (logits, embeddings) при обучении;
            embeddings при инференсе
        """
        # Прямой проход через бэкбон
        outputs = self.backbone(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state[:, 1:]  # без [CLS]
        x = tokens.transpose(1, 2)                 # B×hidden×seq_len
        x = self.pool(x).squeeze(-1)               # B×hidden

        # Проекция и нормализация эмбеддинга
        emb = F.normalize(self.embedding(x), dim=1)
        emb = self.dropout(emb)

        if labels is not None:
            # Обучение: вычисляем логиты через ArcFace
            logits = self.arcface(emb, labels)
            return logits, emb
        # Инференс: возвращаем эмбеддинг
        return emb


class ArcFaceLoss(nn.Module):
    """
    Реализация ArcFace loss: margin-based классификатор.
    """
    def __init__(self, in_features: int, out_features: int, s: float = 10.0, m: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        # Инициализация весов
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Margin и подготовка буферов
        self.m = m
        self.register_buffer('cos_m', torch.cos(torch.tensor(self.m)))
        self.register_buffer('sin_m', torch.sin(torch.tensor(self.m)))
        self.register_buffer('th', torch.cos(torch.tensor(pi - self.m)))
        self.register_buffer('mm', torch.sin(torch.tensor(pi - self.m)) * self.m)

    def set_margin(self, new_m: float):
        """
        Динамически обновить margin.
        """
        self.m = new_m
        self.cos_m.copy_(torch.cos(torch.tensor(self.m)))
        self.sin_m.copy_(torch.sin(torch.tensor(self.m)))
        self.th.copy_(torch.cos(torch.tensor(pi - self.m)))
        self.mm.copy_(torch.sin(torch.tensor(pi - self.m)) * self.m)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            embeddings: Tensor[B, in_features]
            labels:     Tensor[B]
        Returns:
            логиты Tensor[B, out_features]
        """
        # Вычисляем cosinus угла
        cosine = F.linear(F.normalize(embeddings, dim=1), F.normalize(self.weight, dim=1))
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=0.0))
        # Добавляем margin
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Одна «горячая» матрица для true классов
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Итоговые логиты
        output = one_hot * phi + (1.0 - one_hot) * cosine
        return output * self.s