import cv2

video_path = "videos/vehicle-counting.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Could not open video at {video_path}")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Video Frame', frame)

        # 'q' 키를 누르면 비디오 재생이 중지됩니다.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
