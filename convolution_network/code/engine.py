import time
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()

    train_loss = 0
    all_preds = []
    all_labels = []

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        all_preds.extend(y_pred_class.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    train_loss = train_loss / len(dataloader)
    train_acc = accuracy_score(all_labels, all_preds)
    train_precision = precision_score(
        all_labels, all_preds, average="macro", zero_division=0
    )
    train_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    train_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return train_loss, train_acc, train_precision, train_recall, train_f1


def test_step(model, dataloader, loss_fn, device):
    model.eval()

    test_loss = 0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            all_preds.extend(y_pred_class.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    test_loss = test_loss / len(dataloader)
    test_acc = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(
        all_labels, all_preds, average="macro", zero_division=0
    )
    test_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return test_loss, test_acc, test_precision, test_recall, test_f1


def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs, device):
    writer = SummaryWriter("log/regression")

    results = {
        "train_loss": [],
        "train_acc": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "test_loss": [],
        "test_acc": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
    }

    for epoch in range(epochs):
        epoch_start = time.time()

        train_loss, train_acc, train_precision, train_recall, train_f1 = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        test_loss, test_acc, test_precision, test_recall, test_f1 = test_step(
            model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
        )

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds\n"
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"train_precision: {train_precision:.4f} | "
            f"train_recall: {train_recall:.4f} | "
            f"train_f1: {train_f1:.4f}\n"
            f"test_loss : {test_loss:.4f} | "
            f"test_acc : {test_acc:.4f} | "
            f"test_precision : {test_precision:.4f} | "
            f"test_recall : {test_recall:.4f} | "
            f"test_f1 : {test_f1:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_precision"].append(train_precision)
        results["train_recall"].append(train_recall)
        results["train_f1"].append(train_f1)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["test_precision"].append(test_precision)
        results["test_recall"].append(test_recall)
        results["test_f1"].append(test_f1)

        writer.add_scalar("train loss", train_loss, epoch)
        writer.add_scalar("train acc", train_acc, epoch)
        writer.add_scalar("train precision", train_precision, epoch)
        writer.add_scalar("train recall", train_recall, epoch)
        writer.add_scalar("train f1", train_f1, epoch)
        writer.add_scalar("test loss", test_loss, epoch)
        writer.add_scalar("test acc", test_acc, epoch)
        writer.add_scalar("test precision", test_precision, epoch)
        writer.add_scalar("test recall", test_recall, epoch)
        writer.add_scalar("test f1", test_f1, epoch)

    return results
