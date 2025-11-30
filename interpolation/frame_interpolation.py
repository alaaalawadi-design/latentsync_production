import cv2
import torch
from torch.nn import functional as F
from train_log.RIFE_HDv3 import Model
import warnings
warnings.filterwarnings("ignore")

class FrameIntrpolation:
    def __init__(self, chk_path='./train_log'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        self.model = Model()
        self.model.load_model(chk_path, -1)
        self.model.eval()
        self.model.device()

    def execute(self, first_frame, last_frame, fps, save_path):
        img0 = (torch.tensor(first_frame.transpose(2, 0, 1)).to(self.device) / 255.).unsqueeze(0)
        img1 = (torch.tensor(last_frame.transpose(2, 0, 1)).to(self.device) / 255.).unsqueeze(0)
        n, c, h, w = img0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)
        img_list = [img0, img1]
        for i in range(4):
            tmp = []
            for j in range(len(img_list) - 1):
                mid = self.model.inference(img_list[j], img_list[j + 1])
                tmp.append(img_list[j])
                tmp.append(mid)
            tmp.append(img1)
            img_list = tmp
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        video = cv2.VideoWriter(save_path, fourcc, fps, (w, h))        
        for i in range(1, len(img_list[:-1])):   
            frame = (img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
            video.write(frame) 
        video.release() 
        

        
if __name__ == '__main__':
    model = FrameIntrpolation('./train_log')
    img1 = '/media/ehab/46EEC3E77E2602C6/Cyshield/video-background-removal/videos/frames/596.png'
    img2 = '/media/ehab/46EEC3E77E2602C6/Cyshield/video-background-removal/videos/frames/1.png'
    model.execute(cv2.imread(img1), cv2.imread(img2), 10, '/media/ehab/46EEC3E77E2602C6/Cyshield/video-background-removal/videos/ref_video_connection.mp4')