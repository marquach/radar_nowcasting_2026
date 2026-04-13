import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pysteps.visualization as pyvis


from pilot.create_dataset import rain_precipitation_mch
ds = rain_precipitation_mch('pilot/data', window_size=7, target_size=1)

batch_size = 7

train_ratio = 0.8
validation_ratio = 0.2

dataset_size = ds.__len__()
train_size = int(train_ratio * dataset_size)
validation_size = dataset_size - train_size

train_data, test_data = random_split(ds, [train_size, validation_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

from pilot.model.u_net import UNet
model = UNet(batch_size,1)

def train_epoch(epoch_index):
    model.train()
    running_loss = 0.
    num_batches = 0
    

    for i, data in enumerate(train_loader):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)

         # Compute the loss and its gradients
        weight = torch.where(labels > 0, 10.0, 1.0)
        loss = (weight * (outputs - labels) ** 2).mean()
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        num_batches += 1
    epoch_loss = running_loss/num_batches

    return epoch_loss


optimizer  = optim.Adam(model.parameters())


# Initializing in a separate cell so we can easily add more epochs to the same run

epoch_number = 0
EPOCHS = 5

for epoch in range(EPOCHS):
    print(f'EPOCH:{epoch}')

    avg_loss = train_epoch(epoch)

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(validation_loader):
            v_inputs, v_target = data

            v_output = model(v_inputs)
            weight = torch.where(v_target > 0, 10.0, 1.0)
            v_loss = (weight * (v_output - v_target) ** 2).mean()
            #v_loss = mse(v_output, v_target) + 1 * (1-torchmetrics.functional.structural_similarity_index_measure(v_output, v_target))

        v_avgloss = v_loss/(i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, v_avgloss))


import random

random.seed(42)
sample_indices = random.sample(range(len(test_data)), 3)

model.eval()
orig_h, orig_w = ds.data.shape[1], ds.data.shape[2]

fig, axes = plt.subplots(3, 2, figsize=(14, 15))

for row, idx in enumerate(sample_indices):
    x, y = test_data[idx]
    
    with torch.no_grad():
        pred = model(x.unsqueeze(0))
    
    pred_up = F.interpolate(pred, (orig_h, orig_w), mode='bilinear').squeeze()
    y_up = F.interpolate(y.unsqueeze(0), (orig_h, orig_w), mode='bilinear').squeeze()
    
    pred_mm = ds.inverse_transform(pred_up)
    y_mm = ds.inverse_transform(y_up)
    
    plt.sca(axes[row, 0])
    pyvis.plot_precip_field(y_mm, axis='on')
    axes[row, 0].set_title(f"Ground Truth (sample {idx})")
    
    plt.sca(axes[row, 1])
    pyvis.plot_precip_field(pred_mm, axis='on')
    axes[row, 1].set_title(f"Prediction (sample {idx})")

plt.tight_layout()
plt.savefig('proof_of_concepts.png')