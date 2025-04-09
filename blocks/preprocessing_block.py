import cv2 

class PreProcessingBlock:
    def __init__(self):
        pass

    def get_first_last_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, first_frame = cap.read()
        if not ret:
            print("Failed to read the first frame.")
            return None
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, last_frame = cap.read()
        if not ret:
            print("Failed to read the last frame.")
            return None
        cap.release()
        return first_frame, last_frame