import os
import cv2

# 이미지 파일 경로 리스트
image_files = [
    '/workspace/bytetracker/datasets/HT21/HT21-03/img1/000350.jpg',
    '/workspace/bytetracker/datasets/HT21/HT21-03/img1/000390.jpg',
    '/workspace/bytetracker/datasets/HT21/HT21-04/img1/000604.jpg',
    '/workspace/bytetracker/datasets/HT21/HT21-02/img1/001577.jpg'
]

def check_image_file(image_path):
    # 파일 존재 여부 확인
    if not os.path.exists(image_path):
        print(f"파일이 존재하지 않습니다: {image_path}")
        return
    
    # 파일 읽기 시도
    img = cv2.imread(image_path)
    if img is None:
        print(f"이미지 파일을 읽을 수 없습니다: {image_path}")
    else:
        print(f"이미지 파일이 정상적으로 읽혔습니다: {image_path}, 크기: {img.shape}")

# 각 이미지 파일을 확인
for image_file in image_files:
    check_image_file(image_file)
