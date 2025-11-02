import open3d as o3d
import numpy as np
# pyorbbecsdk 임포트 (가정)
from pyorbbecsdk import Pipeline, Config 

# 1. Open3D 비차단(non-blocking) 시각화기 초기화
vis = o3d.visualization.Visualizer()
vis.create_window()
pcd = o3d.geometry.PointCloud()
is_first_frame = True

# 2. Orbbec SDK 파이프라인 설정
pipeline = Pipeline()
config = Config()
# ... (카메라 설정) ...
pipeline.start(config)

try:
    while True:
        # 3. Orbbec SDK에서 프레임 및 포인트 클라우드 데이터 가져오기
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            continue
        
        # SDK의 포인트 클라우드 생성 기능을 사용 (API는 문서 참고 필요)
        # 예시: point_frame = point_cloud_filter.process(frames)
        # 예시: points_data = np.asarray(point_frame.get_data()) # (N, 3) 형태의 NumPy 배열
        
        # (임시) NumPy 배열을 직접 생성한다고 가정
        # 실제로는 이 부분을 SDK에서 받은 데이터로 채워야 합니다.
        points_data = np.random.rand(10000, 3) # <-- 이 부분을 실제 데이터로 교체
        
        # 4. Open3D PointCloud 객체 업데이트
        pcd.points = o3d.utility.Vector3dVector(points_data)
        
        # 5. Open3D 뷰어 업데이트
        if is_first_frame:
            vis.add_geometry(pcd)
            is_first_frame = False
        else:
            vis.update_geometry(pcd)
        
        if not vis.poll_events():
            break
        vis.update_renderer()

finally:
    pipeline.stop()
    vis.destroy_window()