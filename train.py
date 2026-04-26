import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import HighResDataset
from model import CompactSRResNet

# config
DATASET_PATH = "/Users/meheran/Downloads/4K Images Archive"
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4


def main():
    # hardware setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"using device: {device}")

    # load data
    print("loading 4k dataset...")
    dataset = HighResDataset(image_dir=DATASET_PATH, crop_size=96, scale_factor=4)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # init model
    model = CompactSRResNet(scale_factor=4).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # training loop
    print("starting training...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for batch_idx, (lr_imgs, hr_imgs) in enumerate(dataloader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)

            optimizer.zero_grad()
            outputs = model(lr_imgs)

            loss = criterion(outputs, hr_imgs)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"epoch [{epoch + 1}/{EPOCHS}] batch [{batch_idx}/{len(dataloader)}] loss: {loss.item():.4f}")

        # save checkpoint
        torch.save(model.state_dict(), "compact_srresnet.pth")
        print(f"epoch {epoch + 1} complete. model saved.")


if __name__ == "__main__":
    main()