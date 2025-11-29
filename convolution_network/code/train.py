import time
import torch
import data_setup, engine, model, utils

NUM_EPOCHS = 200
BATCH_SIZE = 512
LEARNING_RATE = 0.001

train_dir = "data"
test_dir = "data"

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir, test_dir=test_dir, batch_size=BATCH_SIZE
)

model = model.MyNetwork(input_channels=3, output_channels=len(class_names)).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

start_time = time.time()
engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    epochs=NUM_EPOCHS,
    device=device,
)
total_time = time.time() - start_time
print(f"Total training time: {total_time:.2f} seconds")

utils.save_model(
    model=model,
    target_dir="models",
    model_name="my_model.pth",
)
