import torch
import torchvision
import torchvision.transforms as transforms


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