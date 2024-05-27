import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# 데이터 변환 정의
transform = transforms.Compose([
    transforms.ToTensor(),  # 이미지를 PyTorch 텐서로 변환
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10의 평균과 표준편차로 정규화
])

# CIFAR-10 학습 데이터셋 불러오기
trainset = torchvision.datasets.CIFAR10(
    root='./data',  # 데이터셋을 저장할 경로
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

# CIFAR-10 테스트 데이터셋 불러오기
testset = torchvision.datasets.CIFAR10(
    root='./data',
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

# 데이터셋 클래스 이름 정의
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 데이터셋 로드 확인
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
    print(' '.join('%5s' % classes[labels[j]] for j in range(8)))