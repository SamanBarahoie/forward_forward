import torch
from model import FF_Network
from data_loader import get_mnist
import torch.nn.functional as F
import torch.optim as optim
import random

# Create negative examples by shuffling labels
def create_negative_samples(x, y):
    y_neg = y[torch.randperm(len(y))]
    return x, y_neg

def label_to_one_hot(y, num_classes=10):
    return F.one_hot(y, num_classes=num_classes).float()

def train_ff():
    train_loader, _ = get_mnist()
    input_dim = 28 * 28 + 10
    model = FF_Network([input_dim, 500, 500])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        total_loss = 0
        for x, y in train_loader:
            x = x.view(x.size(0), -1)
            x_pos = torch.cat([x, label_to_one_hot(y)], dim=1)
            _, y_neg = create_negative_samples(x, y)
            x_neg = torch.cat([x, label_to_one_hot(y_neg)], dim=1)

            activations_pos = model.forward_pass(x_pos)
            activations_neg = model.forward_pass(x_neg)

            loss = 0
            for ap, an, layer in zip(activations_pos, activations_neg, model.layers):
                gp = layer.goodness(ap)
                gn = layer.goodness(an)
                l = torch.log(1 + torch.exp(-gp + 2.0)).mean() + torch.log(1 + torch.exp(gn - 2.0)).mean()
                loss += l

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "ff_model.pt")

if __name__ == '__main__':
    train_ff()
