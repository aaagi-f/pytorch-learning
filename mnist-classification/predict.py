import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# ========== 网络定义（必须和训练时一样）==========
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# ========== 加载模型 ==========
net = Net()
net.load_state_dict(torch.load('best_mnist_model.pth'))
net.eval()

# ========== 预处理 ==========
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ========== 预测函数 ==========
def predict_image(image_path):
    image = Image.open(image_path).convert('L')
    image = image.resize((28, 28))
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = net(image_tensor)
        _, predicted = torch.max(output.data, 1)
    
    return predicted.item()

# ========== 测试：用MNIST测试集第0张 ==========
testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

img, label = testset[0]
img_batch = img.unsqueeze(0)

with torch.no_grad():
    output = net(img_batch)
    _, predicted = torch.max(output.data, 1)

print(f'测试集第0张图，真实标签: {label}，模型预测: {predicted.item()}')

# 用自己的图时取消下面这行注释：
# print(predict_image('my_3.png'))