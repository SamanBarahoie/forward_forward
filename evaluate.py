import torch
from model import FF_Network
from data_loader import get_mnist
from train import label_to_one_hot

def evaluate():
    _, test_loader = get_mnist()
    input_dim = 28 * 28 + 10
    model = FF_Network([input_dim, 500, 500])
    model.load_state_dict(torch.load("ff_model.pt"))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.view(x.size(0), -1)
            scores = []
            for label in range(10):
                label_tensor = torch.full_like(y, label)
                x_test = torch.cat([x, label_to_one_hot(label_tensor)], dim=1)
                acts = model.forward_pass(x_test)
                goodness = sum([layer.goodness(act) for layer, act in zip(model.layers, acts)])
                scores.append(goodness.unsqueeze(1))
            scores = torch.cat(scores, dim=1)
            preds = scores.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")
evaluate()