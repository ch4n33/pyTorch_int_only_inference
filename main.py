import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model import *
from util import *


g_test = False
def main():
    print('데이터 변환 정의') 
    transform = transforms.Compose([
        transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))  # CIFAR-10의 평균과 표준편차로 정규화
    ])

    print('CIFAR-10 학습 데이터셋 불러오기')
    trainset = torchvision.datasets.CIFAR10(
        root='./CIFAR10_data',  # 데이터셋을 저장할 경로
        train=True,  # 학습용 데이터셋
        download=True,  # 데이터셋이 경로에 없으면 다운로드
        transform=transform  # 정의한 변환 적용
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=64,  # 미니배치 크기
        shuffle=True,  # 데이터를 섞어서 로드
        num_workers=2  # 데이터 로드에 사용할 프로세스 수
    )

    print('CIFAR-10 테스트 데이터셋 불러오기')
    testset = torchvision.datasets.CIFAR10(
        root='./CIFAR10_data',
        train=False,  # 테스트용 데이터셋
        download=True,
        transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )

    print('데이터셋 클래스 이름 정의')
    cifar_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('데이터셋 로드 확인')
    def loadtest():
        # 이미지를 보여주는 함수 정의
        def imshow(img):
            img = img / 2 + 0.5  # 정규화 해제
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()
        
        # 학습용 데이터셋에서 일부 이미지 가져오기
        dataiter = iter(trainloader)
        images, labels = dataiter.next()

        # 이미지 보여주기
        imshow(torchvision.utils.make_grid(images))
        # 레이블 출력
        print(' '.join('%5s' % cifar_classes[labels[j]] for j in range(8)))
    if g_test:
        loadtest()

    # dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    # torch.device(dev)
    # print('학습을 위한 장치:', dev)

    def train(model, crit, opt, epoch=2, show=False):
        for epoch in range(epoch):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                opt.zero_grad()
                # print(inputs.shape, labels.shape) : torch.Size([64, 3, 32, 32]) torch.Size([64])
                outputs = model(inputs)
                loss = crit(outputs, labels)
                loss.backward()
                opt.step()
                running_loss += loss.item()
                r = 100
                if i % r == r-1:
                    print(f'[{epoch + 1}, {i + 1}] loss: {(running_loss / r):.3f}')
                    running_loss = 0.0

    def validate(model):
        correct = 0
        total = 0
        prediction_statistics = np.zeros((10, 10))
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # update prediction statistics
                for i in range(len(labels)):
                    prediction_statistics[labels[i]][predicted[i]] += 1
        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')
        return prediction_statistics
    criterion = nn.CrossEntropyLoss()
    model_no_qat = MLP(hidden_dim=256)
    optimizer_no_qat = torch.optim.SGD(model_no_qat.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    
    print('모델 학습 시작')
    train(model_no_qat, criterion, optimizer_no_qat, 1)
    prediction_matrix = validate(model_no_qat)
    print('Finished Training')
    print('Confusion matrix\n', prediction_matrix)
    
    model_qat = MLP(hidden_dim=256)
    
    optimizer_qat = torch.optim.SGD(model_qat.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-3)
    
    print('모델 QAT 시작')
    with QuantizationEnabler(model_qat): # type: ignore
        train(model_qat, criterion, optimizer_qat, 1, show=True)
    print('Finished QTA')
    prediction_matrix = validate(model_qat) 

    print('Confusion matrix\n', prediction_matrix)
    
    #plot the scales
    max_vals = model_qat.get_max_vals()
    scales = model_qat.get_scales()
    means = model_qat.get_means()
    bias_means = model_qat.get_bias_means()

    # fig, ax = plt.subplots(3, 1, figsize=(5, 6), constrained_layout=True)
    # for i in range(len(max_vals)):
    #     ax[0].plot(max_vals[i], label=f'Layer {i}')
    #     ax[1].plot(scales[i], label=f'Layer {i}')
    #     ax[2].plot(means[i], label=f'Layer {i}')
    #     ax[2].plot(bias_means[i], label=f'Layer {i}', linestyle='--')
    # ax[0].set_xlabel('Iteration')
    # ax[0].set_ylabel('Max Value')
    # ax[0].legend()
    # ax[1].set_xlabel('Iteration')
    # ax[1].set_ylabel('Scale')
    # ax[1].legend()  
    # ax[2].set_xlabel('Iteration')
    # ax[2].set_ylabel('Mean')
    # ax[2].legend()
    mx = model_qat.get_rangetrackers_max()
    mn = model_qat.get_rangetrackers_min()
    fig, ax = plt.subplots(2, 1, figsize=(5, 6), constrained_layout=True)
    for i in range(len(max_vals)):
        ax[0].plot(mx[i], label=f'Layer {i}')
        ax[1].plot(mn[i], label=f'Layer {i}')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Max Value')
    ax[0].legend()
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('Min Value')
    ax[1].legend()
    
    plt.show()

if __name__ == '__main__':
    main()