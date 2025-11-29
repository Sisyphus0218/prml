import torch
import data_setup, engine, model

BATCH_SIZE = 128

train_dir = "data"
test_dir = "data"

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir, test_dir=test_dir, batch_size=BATCH_SIZE
)

model = model.MyNetwork(input_channels=3, output_channels=len(class_names)).to(device)

model.load_state_dict(torch.load("models/my_model.pth", weights_only=False))
model.to(device)

loss_fn = torch.nn.CrossEntropyLoss()

test_loss, test_acc, test_precision, test_recall, test_f1 = engine.test_step(
    model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device
)

print(
    f"test_loss : {test_loss:.4f} | "
    f"test_acc : {test_acc:.4f} | "
    f"test_precision : {test_precision:.4f} | "
    f"test_recall : {test_recall:.4f} | "
    f"test_f1 : {test_f1:.4f}"
)
