import os 
import pickle 

class ReferenceVideoManager:
    def __init__(self, base_folder):
        self.base_folder = base_folder
        self.ref_video_versions = self._find_versions()

    def _find_versions(self):
        # Find all version folders dynamically
        return sorted([
            d for d in os.listdir(self.base_folder)
            if os.path.isdir(os.path.join(self.base_folder, d))
        ])

    def get_videos_for_version(self, version_name):
        version_path = os.path.join(self.base_folder, version_name)
        ref_video = os.path.join(version_path, "ref_video.mp4")
        silent_video = os.path.join(version_path, "silent.mp4")
        return ref_video, silent_video
    
    def set_ref_video(self, version):
        ref_pkl = os.path.join(self.base_folder, version, 'ref_frames_data.pkl')
        with open(ref_pkl, "rb") as f:
            out_video_with_scielnt = pickle.load(f)
            all_faces, all_original_video_frames, all_boxes, all_affine_matrices = out_video_with_scielnt
        ref_video_data = {'faces': all_faces,
                          'original_video_frames': all_original_video_frames,
                          'boxes': all_boxes,
                          'affine_matrices': all_affine_matrices
                          }
        return ref_video_data
