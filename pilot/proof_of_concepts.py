import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pysteps.visualization as pyvis

from pilot.create_dataset import RainPrecipitationMCH
from pilot.model.u_net import UNet

WINDOW_SIZE = 7
TARGET_SIZE = 1
BATCH_SIZE = 7
TRAIN_RATIO = 0.8
EPOCHS = 5


def train_epoch(model, optimizer, train_loader):
    model.train()
    running_loss = 0.0
    num_batches = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)

        weight = torch.where(labels > 0, 10.0, 1.0)
        loss = (weight * (outputs - labels) ** 2).mean()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches


def validate(model, validation_loader):
    model.eval()
    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for v_inputs, v_target in validation_loader:
            v_output = model(v_inputs)
            weight = torch.where(v_target > 0, 10.0, 1.0)
            v_loss = (weight * (v_output - v_target) ** 2).mean()
            running_loss += v_loss.item()
            num_batches += 1

    return running_loss / num_batches


def plot_predictions(model, ds, test_data, save_path="proof_of_concepts.png"):
    random.seed(42)
    sample_indices = random.sample(range(len(test_data)), 3)

    model.eval()
    orig_h, orig_w = ds.data.shape[1], ds.data.shape[2]

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    for row, idx in enumerate(sample_indices):
        x, y = test_data[idx]

        with torch.no_grad():
            pred = model(x.unsqueeze(0))

        pred_up = F.interpolate(pred, (orig_h, orig_w), mode="bilinear").squeeze()
        y_up = F.interpolate(y.unsqueeze(0), (orig_h, orig_w), mode="bilinear").squeeze()

        pred_mm = ds.inverse_transform(pred_up)
        y_mm = ds.inverse_transform(y_up)

        plt.sca(axes[row, 0])
        pyvis.plot_precip_field(y_mm, axis="on")
        axes[row, 0].set_title(f"Ground Truth (sample {idx})")

        plt.sca(axes[row, 1])
        pyvis.plot_precip_field(pred_mm, axis="on")
        axes[row, 1].set_title(f"Prediction (sample {idx})")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved predictions to {save_path}")


if __name__ == "__main__":
    ds = RainPrecipitationMCH("pilot/data", window_size=WINDOW_SIZE, target_size=TARGET_SIZE)

    dataset_size = len(ds)
    train_size = int(TRAIN_RATIO * dataset_size)
    validation_size = dataset_size - train_size

    train_data, test_data = random_split(
        ds, [train_size, validation_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    model = UNet(WINDOW_SIZE, 1)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(EPOCHS):
        avg_loss = train_epoch(model, optimizer, train_loader)
        v_avgloss = validate(model, validation_loader)
        print(f"EPOCH {epoch}  train_loss={avg_loss:.4f}  val_loss={v_avgloss:.4f}")

    torch.save(model.state_dict(), "pilot/model/unet.pth")
    plot_predictions(model, ds, test_data)