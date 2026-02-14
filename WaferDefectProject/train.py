import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.dataset import WaferMapDataset
from src.model import get_mobilenetv4


def parse_args():
    project_root = Path(__file__).resolve().parent
    default_data = project_root.parent / "dataset_repo" / "Wafer_Map_Datasets.npz"
    parser = argparse.ArgumentParser(description="Wafer defect training")
    parser.add_argument("--data-path", type=str, default=str(default_data))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default="", help="Path to training checkpoint for resume")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    return parser.parse_args()


def train():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    is_windows = os.name == "nt"

    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    gpu_count = torch.cuda.device_count() if device == "cuda" else 0
    print(f"Using device: {device}, GPUs: {gpu_count}")
    print(f"Dataset path: {args.data_path}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = WaferMapDataset(args.data_path, mode="train", transform=train_transform)
    val_ds = WaferMapDataset(args.data_path, mode="test", transform=val_transform)

    pin_memory = device == "cuda"
    worker_count = max(0, args.num_workers)
    persistent_workers = (worker_count > 0) and (not is_windows)
    print(
        f"Runtime config | batch_size={args.batch_size}, num_workers={worker_count}, "
        f"windows={is_windows}, pin_memory={pin_memory}, persistent_workers={persistent_workers}"
    )

    loader_kwargs = dict(
        num_workers=worker_count,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    if worker_count > 0 and not is_windows:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    model = get_mobilenetv4(num_classes=train_ds.num_classes, pretrained=args.pretrained).to(device)
    if gpu_count > 1:
        model = nn.DataParallel(model)
        print("Enabled nn.DataParallel for multi-GPU training")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best_acc = 0.0
    target_acc = 97.2
    start_epoch = 0

    if args.resume:
        if os.path.exists(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            ckpt = torch.load(args.resume, map_location="cpu")
            state_dict = ckpt["model"] if "model" in ckpt else ckpt
            model_to_load = model.module if isinstance(model, nn.DataParallel) else model
            model_to_load.load_state_dict(state_dict)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
            if "scaler" in ckpt and device == "cuda":
                scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt.get("epoch", 0)
            best_acc = ckpt.get("best_acc", 0.0)
            print(f"Resume state: start_epoch={start_epoch}, best_acc={best_acc:.2f}%")
        else:
            print(f"Resume checkpoint not found: {args.resume}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                {
                    "loss": f"{train_loss / max(1, (pbar.n + 1)):.4f}",
                    "acc": f"{100.0 * correct / max(1, total):.2f}%",
                }
            )

        scheduler.step()

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / max(1, total)
        val_acc = 100.0 * val_correct / max(1, val_total)
        epoch_time = time.time() - start_time
        train_throughput = total / max(epoch_time, 1e-6)
        print(
            f"Epoch {epoch + 1:03d} | Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}% | Time: {epoch_time:.1f}s | "
            f"Train Throughput: {train_throughput:.1f} img/s"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, os.path.join(args.save_dir, "mobilenetv4_best.pth"))
            print("Saved Best Model!")

        latest_ckpt = {
            "epoch": epoch + 1,
            "best_acc": best_acc,
            "model": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if device == "cuda" else None,
            "args": vars(args),
        }
        torch.save(latest_ckpt, os.path.join(args.save_dir, "mobilenetv4_latest.pth"))

        if best_acc >= target_acc:
            print(f"Target reached: Best Acc {best_acc:.2f}% >= {target_acc:.1f}%")

    final_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(final_state, os.path.join(args.save_dir, "mobilenetv4_final.pth"))
    print(f"Training Complete. Best Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    train()
