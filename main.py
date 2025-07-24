import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.utils.parametrize as parametrize
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set deterministic behavior
_ = torch.manual_seed(0)

# Define data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST train and test sets
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)

mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=10, shuffle=True)

# Define the MLP model (formerly RichBoyNet)
class MNISTDenseClassifier(nn.Module):
    def __init__(self, hidden_size_1=1000, hidden_size_2=2000):
        super(MNISTDenseClassifier, self).__init__()
        self.linear1 = nn.Linear(28*28, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, 10)
        self.relu = nn.ReLU()

    def forward(self, img):
        x = img.view(-1, 28*28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

net = MNISTDenseClassifier().to(device)

# Training function
def train(train_loader, net, epochs=5, total_iterations_limit=None):
    cross_el = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    total_iterations = 0

    for epoch in range(epochs):
        net.train()
        loss_sum = 0
        num_iterations = 0
        data_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}')

        if total_iterations_limit is not None:
            data_iterator.total = total_iterations_limit

        for data in data_iterator:
            num_iterations += 1
            total_iterations += 1
            x, y = data
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = net(x)
            loss = cross_el(output, y)
            loss_sum += loss.item()
            avg_loss = loss_sum / num_iterations
            data_iterator.set_postfix(loss=avg_loss)
            loss.backward()
            optimizer.step()

            if total_iterations_limit is not None and total_iterations >= total_iterations_limit:
                return

# Save original weights
original_weights = {}
for name, param in net.named_parameters():
    original_weights[name] = param.clone().detach()

# Testing function
def test():
    correct = 0
    total = 0
    wrong_counts = [0 for _ in range(10)]

    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing'):
            x, y = data
            x, y = x.to(device), y.to(device)
            output = net(x)
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                else:
                    wrong_counts[y[idx]] += 1
                total += 1

    print(f'Accuracy: {round(correct/total, 3)}')
    for i in range(len(wrong_counts)):
        print(f'Wrong counts for digit {i}: {wrong_counts[i]}')

# Evaluate baseline performance
test()

# Print model parameter sizes
total_parameters_original = 0
for index, layer in enumerate([net.linear1, net.linear2, net.linear3]):
    total_parameters_original += layer.weight.nelement() + layer.bias.nelement()
    print(f'Layer {index+1}: W: {layer.weight.shape} + B: {layer.bias.shape}')
print(f'Total parameters: {total_parameters_original:,}')

# Define LoRA parameterization
class LoRAParametrization(nn.Module):
    def __init__(self, features_in, features_out, rank=1, alpha=1, device='cpu'):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros((rank, features_out)).to(device))
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))
        nn.init.normal_(self.lora_A, mean=0, std=1)
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, original_weights):
        if self.enabled:
            return original_weights + torch.matmul(self.lora_B, self.lora_A).view(original_weights.shape) * self.scale
        else:
            return original_weights

def linear_layer_parameterization(layer, device, rank=1, lora_alpha=1):
    features_in, features_out = layer.weight.shape
    return LoRAParametrization(features_in, features_out, rank=rank, alpha=lora_alpha, device=device)

# Register LoRA to all layers
parametrize.register_parametrization(net.linear1, "weight", linear_layer_parameterization(net.linear1, device))
parametrize.register_parametrization(net.linear2, "weight", linear_layer_parameterization(net.linear2, device))
parametrize.register_parametrization(net.linear3, "weight", linear_layer_parameterization(net.linear3, device))

# Enable or disable LoRA
def enable_disable_lora(enabled=True):
    for layer in [net.linear1, net.linear2, net.linear3]:
        layer.parametrizations["weight"][0].enabled = enabled

# Count parameters introduced by LoRA
total_parameters_lora = 0
total_parameters_non_lora = 0
for index, layer in enumerate([net.linear1, net.linear2, net.linear3]):
    total_parameters_lora += layer.parametrizations["weight"][0].lora_A.nelement() + layer.parametrizations["weight"][0].lora_B.nelement()
    total_parameters_non_lora += layer.weight.nelement() + layer.bias.nelement()
    print(f'Layer {index+1}: W: {layer.weight.shape} + B: {layer.bias.shape} + LoRA_A: {layer.parametrizations["weight"][0].lora_A.shape} + LoRA_B: {layer.parametrizations["weight"][0].lora_B.shape}')

assert total_parameters_non_lora == total_parameters_original
print(f'Total parameters (original): {total_parameters_non_lora:,}')
print(f'Total parameters (original + LoRA): {total_parameters_lora + total_parameters_non_lora:,}')
print(f'Parameters introduced by LoRA: {total_parameters_lora:,}')
print(f'Parameters increment: {(total_parameters_lora / total_parameters_non_lora) * 100:.3f}%')

# Freeze non-LoRA parameters
for name, param in net.named_parameters():
    if 'lora' not in name:
        param.requires_grad = False

# Fine-tune only on digit 9
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
only_nine = mnist_trainset.targets == 9
mnist_trainset.data = mnist_trainset.data[only_nine]
mnist_trainset.targets = mnist_trainset.targets[only_nine]
train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=10, shuffle=True)

train(train_loader, net, epochs=1, total_iterations_limit=100)

# Validate original weights remain unchanged
assert torch.all(net.linear1.parametrizations.weight.original == original_weights['linear1.weight'])
assert torch.all(net.linear2.parametrizations.weight.original == original_weights['linear2.weight'])
assert torch.all(net.linear3.parametrizations.weight.original == original_weights['linear3.weight'])

# Test with LoRA enabled
enable_disable_lora(enabled=True)
test()

# Test with LoRA disabled
enable_disable_lora(enabled=False)
test()
