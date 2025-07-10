import os
import json
import torch
from torch.optim import AdamW
# from transformers import get_linear_schedule_with_warmup
from transformers import get_cosine_schedule_with_warmup

from model import DiT_FN
from preprocess import train_loader, val_loader, test_loader

def train(
    model,
    train_loader,
    val_loader,
    device,
    optimizer,
    scheduler,
    num_epochs,
    accum_steps,
    output_dir="checkpoints"
):
    """
    Обучение модели с warm-up и сохранением лучших чекпоинтов.

    Args:
        model: PyTorch модель с forward, возвращающим (logits, embeddings)
        train_loader, val_loader: DataLoader для train и val
        device: устройство ('cpu' или 'cuda')
        optimizer: оптимизатор AdamW
        scheduler: lr scheduler с warm-up
        num_epochs: число эпох
        output_dir: директория для чекпоинтов и конфига
    """

    # Подготовка директории и сохранение исходных гиперпараметров
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    hparams = {
        "device": str(device),
        "num_epochs": num_epochs,
        "optimizer": type(optimizer).__name__,
        "lr": optimizer.defaults.get('lr', None),
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "warmup_steps": getattr(scheduler, 'num_warmup_steps', None),
        "total_steps": getattr(scheduler, 'num_training_steps', None)
    }
    with open(config_
              path, "w") as f:
        json.dump(hparams, f, indent=2)
    
    # подготовка, гиперпараметры, criterion, scaler и т. п.
    start_epoch   = 1
    best_val_loss = float('inf')

    # попытка загрузить last_checkpoint.pt
    ckpt_path = os.path.join(output_dir, "last_checkpoint.pt")
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(   ckpt["model_state"]    )
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        # перенос буферов оптимизатора на GPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        best_val_loss = ckpt["best_val_loss"]
        start_epoch   = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    
    # loss
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)

    # accum_steps x bacth_size = real_batch_size
    scaler = torch.cuda.amp.GradScaler()  # экономит 40 % памяти
    accum_steps = accum_steps  # 4   
    best_val_loss = float('inf')
    best_epoch = 0

    # early_stop
    patience      = 10          # сколько эпох ждём улучшения
    epochs_no_improve = 0       # счётчик «плохих» эпох

    # Цикл по эпохам
    for epoch in range(start_epoch, num_epochs + 1):
        # Обучение
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        # 1) Эпохи 1–2: тренируем только голову
        if epoch == 1:  # 1
            model.unfreeze_backbone(n_last_layers=0)
        
        # 2) Эпохи 3–6: размораживаем последний блок энкодера
        if epoch == 3:  # 3
            model.unfreeze_backbone(n_last_layers=1)
            
        # 3) Эпохи 7–10: ещё по два блока
        if epoch == 7:  # 7
            model.unfreeze_backbone(n_last_layers=2)

        # 4) размораживаем всё
        if epoch == 10:  # 10
            model.unfreeze_backbone(n_last_layers=len(model.backbone.encoder.layer))
           
        model.train()
        optimizer.zero_grad()
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():       # FP16-forward
                logits, _ = model(images, labels)
                loss = criterion(logits, labels) / accum_steps  # делим!

            scaler.scale(loss).backward()
            
            # каждые accum_steps обновляем веса
            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()  # LR-scheduler

            # статистика
            train_loss += loss.item() * accum_steps * labels.size(0)  # умножаем назад
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= train_total
        train_acc   = train_correct / train_total
        
        # Валидация
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits, _ = model(images, labels)
                loss = criterion(logits, labels)
                val_loss += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Сохраняем лучший чекпоинт
        if val_loss < best_val_loss - 2e-5:   # 2e-5 - маленький порог, чтобы игнорировать шум
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0             # сбрасываем счётчик
            torch.save(model.state_dict(), os.path.join(output_dir, "model_best.pth"))
            hparams.update({"best_epoch": best_epoch, 
                            "best_val_loss": best_val_loss, 
                            "best_val_acc": val_acc, 
                            "best_train_acc": train_acc})
            
            with open(config_path, "w") as f:
                json.dump(hparams, f, indent=2)
        else:
            epochs_no_improve += 1             # ухудшилась или стагнация, увеличиваем счётчик
            
            if epochs_no_improve >= patience:
                print(f"Early stop: {patience} эпох без улучшения.")
                break
        
        # Каждые 7 эпох сохраняем чекпоинт      
        if epoch % 7 == 0:
            ckpt = {
                        "epoch":      epoch,
                        "model_state":      model.state_dict(),
                        "optimizer_state":  optimizer.state_dict(),
                        "scheduler_state":  scheduler.state_dict(),
                        "best_val_loss":    best_val_loss,
                        "hparams":          hparams,
                    }
            torch.save(ckpt, "checkpoints/last_checkpoint.pt")
    
    # Сохраняем финальную модель
    torch.save(model.state_dict(), os.path.join(output_dir, "model_final.pth"))
    print(f"Training complete. Best val loss {best_val_loss:.4f} at epoch {best_epoch}.")
    
    return output_dir

# Инициализация и запуск
if __name__ == "__main__":
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_epochs   = 70
    lr_head      = 3e-4
    lr_backbone  = 3e-5
    weight_decay = 1e-3
    lr = 3e-4
    embed_dim=512
    s=15.0
    m=0.15
    drop_p=0.025

    # Инициализация классов
    with open("classes.txt", "r") as f:
        classes = f.read().split()
    
    # Инициализация модели и стратегий
    model = DiT_FN(
        n_classes=len(classes),
        backbone_name="microsoft/dit-base",
        embed_dim=embed_dim,
        s=s,
        m=m,
        drop_p=drop_p
    )
    model.freeze_backbone()

    # optimizer = AdamW(model.parameters(), lr=lr)
    optimizer = AdamW(
    [
        # learning rate для новых слоёв — проекции и ArcFace-головы
        # они инициализируются с нуля и должны быстро войти в задачу, 
        # поэтому им даётся более высокий шаг lr_head
        {'params': model.embedding.parameters(), 'lr': lr_head},  
        {'params': model.arcface.parameters(),  'lr': lr_head},  
        
        # lr_backbone для предобученного бэкбона (DiT)
        # он уже знает, как извлекать признаки, 
        # поэтому докатываем его очень осторожно, маленьким шагом
        {'params': model.backbone.parameters(), 'lr': lr_backbone},
    ],
    weight_decay=weight_decay
    )
    
    # cos scheduler
    accum_steps = 2
    total_steps  = num_epochs * len(train_loader) // accum_steps
    warmup_steps = int(0.12 * total_steps)  # первые 12% шагов — warm-up
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    ## Попытка использовать линейный scheduler
    # # lin scheduler
    # total_steps = num_epochs * len(train_loader)
    # warmup_steps = int(0.1 * total_steps)
    # scheduler = get_linear_schedule_with_warmup(
        # optimizer,
        # num_warmup_steps=warmup_steps,
        # num_training_steps=total_steps
    # )

    # сохранения прамметров для истории
    param_dir = "checkpoints"
    param_path = os.path.join(param_dir, "param.json")
    hparams = {
        "num_epochs": num_epochs,
        "lr_head": lr_head,
        "lr_backbone": lr_backbone,
        "weight_decay": weight_decay,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "embed_dim": embed_dim,
        "s": s,
        "m": m,
        "drop_p": drop_p
    }
    with open(param_path, "w") as f:
        json.dump(hparams, f, indent=2)
    
    print(DEVICE)  # Проверка девайса

    # Запуск
    train(
        model,
        train_loader,
        val_loader,
        DEVICE,
        optimizer,
        scheduler,
        num_epochs,
        accum_steps
    )