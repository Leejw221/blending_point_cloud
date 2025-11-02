# Single Point Cloud - Multi-Camera Point Cloud Blending

Femto-bolt (고정) + D405 (동적) 카메라의 point cloud를 base 좌표계에서 통합하는 프로젝트

## 디렉토리 구조

```
single_point_cloud/
├── core/                   # 핵심 모듈
│   ├── config.py          # 변환행렬 및 설정
│   ├── camera_drivers.py  # 카메라 인터페이스
│   ├── transforms.py      # TF 계산 (PiPER FK 포함)
│   ├── visualizer.py      # Open3D 시각화
│   └── blending.py        # Blending 알고리즘 (예정)
│
├── scripts/               # 실행 스크립트
│   ├── main_visualize.py # 실시간 시각화
│   └── main_blend.py     # Blending 실행 (예정)
│
├── data/                  # 데이터
│   ├── raw/              # 원본 point cloud
│   └── blended/          # Blending 결과
│
└── results/              # 실험 결과
```

## 사용법

### 1. 실시간 시각화

#### 하드웨어 연결 시
```bash
cd /home/leejungwook/point_cloud_blending/single_point_cloud
python3 scripts/main_visualize.py
```

#### 테스트 모드 (하드웨어 없이)
```bash
python3 scripts/main_visualize.py --dummy
```

#### Femto-bolt만 사용
```bash
python3 scripts/main_visualize.py --no-d405
```

#### D405만 사용
```bash
python3 scripts/main_visualize.py --no-femto
```

### 2. Blending (예정)

```bash
python3 scripts/main_blend.py
```

## 필요한 라이브러리

```bash
# 기본 라이브러리
pip3 install numpy scipy open3d

# D405 (아직 설치 안 됨)
pip3 install pyrealsense2

# Femto-bolt (이미 설치됨)
# ../Femto_bolt_calibration/orbbec_sdk에 포함

# PiPER SDK (이미 설치됨)
# piper_ws에서 설치
```

## 좌표계

```
base (reference)
 ├─ femto_bolt_frame (static)
 └─ manipulator_base (static)
     └─ end_effector_frame (dynamic, from PiPER FK)
         └─ d405_frame (static offset)
```

## Transformation Matrices

### Base → Femto-bolt
- File: `../Femto_bolt_calibration/results/depth_to_base_transform.txt`
- Updated: 2024-10-19

### Base → D405
- 계산: `base → manipulator_base → end_effector → d405`
- Dynamic: PiPER FK 사용

## 진행 상황

- [x] 디렉토리 구조 생성
- [x] 카메라 드라이버 구현
- [x] TF 계산 모듈
- [x] Open3D 시각화
- [ ] Point cloud blending 알고리즘
- [ ] 결과 저장 (.npy, .pcd)
- [ ] 여러 시나리오 테스트

## 참고

- Femto-bolt calibration: `../Femto_bolt_calibration/`
- Multi-camera calibration (ROS2): `~/multi-camera_calibration/`
