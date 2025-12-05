# backend/train.py
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from backend.data_prep import get_dataloaders
from backend.model import get_model, save_model
from sklearn.metrics import accuracy_score

def train_loop(root_data_dir, out_weights='backend/weights/best.pt',
               epochs=8, batch_size=16, lr=1e-4, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, _, class_to_idx = get_dataloaders(root_data_dir, batch_size=batch_size)
    num_classes = len(class_to_idx)
    model = get_model(num_classes=num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    best_val_acc = 0.0
    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Train epoch {epoch}"):
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.cpu().numpy().tolist())
        val_acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch}: loss {epoch_loss:.4f} val_acc {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(out_weights), exist_ok=True)
            save_model(model.cpu(), out_weights)
            model.to(device)
            print(f"Saved best model to {out_weights}")

    print("Training complete. Best val acc:", best_val_acc)
    return out_weights, class_to_idx

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="root data dir with train/val/test folders")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", default="backend/weights/best.pt")
    args = parser.parse_args()
    train_loop(args.data, out_weights=args.out, epochs=args.epochs, batch_size=args.batch)
