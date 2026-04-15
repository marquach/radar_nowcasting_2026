import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pysteps.visualization as pyvis

from torchmetrics.functional import structural_similarity_index_measure as ssim

from create_dataset import RainPrecipitationMCH
from model.u_net import UNet

WINDOW_SIZE = 7
TARGET_SIZE = 3
BATCH_SIZE = 256
TRAIN_RATIO = 0.8


EPOCHS = 50
LAMBDA_SSIM = 0.5  # weight for SSIM loss
PATIENCE = 5  # Early stopping patience



def train_epoch(model, optimizer, train_loader):
    model.train()
    running_loss = 0.0
    num_batches = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)

        weight = torch.where(labels > 0, 10.0, 1.0)
        mse_loss = (weight * (outputs - labels) ** 2).mean()
        # SSIM expects (N, C, H, W) in [0, 1]
        ssim_loss = 1 - ssim(outputs, labels, data_range=1.0)
        loss = mse_loss + LAMBDA_SSIM * ssim_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1
        print(f'Loss for batch {num_batches}: {running_loss/num_batches}')
        #

    return running_loss / num_batches



def validate(model, validation_loader):
    model.eval()
    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for v_inputs, v_target in validation_loader:
            v_output = model(v_inputs)
            weight = torch.where(v_target > 0, 10.0, 1.0)
            mse_loss = (weight * (v_output - v_target) ** 2).mean()
            ssim_loss = 1 - ssim(v_output, v_target, data_range=1.0)
            v_loss = mse_loss + LAMBDA_SSIM * ssim_loss
            running_loss += v_loss.item()
            num_batches += 1

    return running_loss / num_batches


def plot_predictions(model, ds, test_data, save_path="proof_of_concepts.png"):
    random.seed(42)
    n_samples = 3
    sample_indices = random.sample(range(len(test_data)), n_samples)

    model.eval()
    orig_h, orig_w = ds.data.shape[1], ds.data.shape[2]
    target_size = ds.target_size

    fig, axes = plt.subplots(n_samples * target_size, 2, figsize=(14, 5 * n_samples * target_size))
    if target_size == 1 and n_samples == 1:
        axes = axes.reshape(1, 2)

    for sample_row, idx in enumerate(sample_indices):
        x, y = test_data[idx]

        with torch.no_grad():
            pred = model(x.unsqueeze(0))

        pred_up = F.interpolate(pred, (orig_h, orig_w), mode="bilinear").squeeze(0)
        y_up = F.interpolate(y.unsqueeze(0), (orig_h, orig_w), mode="bilinear").squeeze(0)

        for t in range(target_size):
            row = sample_row * target_size + t

            pred_mm = ds.inverse_transform(pred_up[t])
            y_mm = ds.inverse_transform(y_up[t])

            plt.sca(axes[row, 0])
            pyvis.plot_precip_field(y_mm, axis="on")
            axes[row, 0].set_title(f"Ground Truth (sample {idx}, t+{t+1})")

            plt.sca(axes[row, 1])
            pyvis.plot_precip_field(pred_mm, axis="on")
            axes[row, 1].set_title(f"Prediction (sample {idx}, t+{t+1})")

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

    model = UNet(WINDOW_SIZE, TARGET_SIZE)
    optimizer = optim.Adam(model.parameters())

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        avg_loss = train_epoch(model, optimizer, train_loader)
        v_avgloss = validate(model, validation_loader)
        print(f"EPOCH {epoch}  train_loss={avg_loss:.4f}  val_loss={v_avgloss:.4f}")

        if v_avgloss < best_val_loss:
            best_val_loss = v_avgloss
            patience_counter = 0
            best_model_state = model.state_dict()
            print("Validation loss improved, saving model.")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs.")
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}.")
                break

    # Save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), "pilot/model/unet_bigger_target_window.pth")
        print("Best model saved.")
    else:
        torch.save(model.state_dict(), "pilot/model/unet_bigger_target_window.pth")

    plot_predictions(model, ds, test_data)