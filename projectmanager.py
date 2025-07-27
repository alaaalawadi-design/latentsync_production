from pathlib import Path
import os 

class ProjectManager:
    def __init__(self, base_dir: Path, models_dir: Path, device="cuda", seed=1247, guidance_scale=2):
        self.base_dir = base_dir.resolve()
        self.models_dir = models_dir.resolve()
        self.device = device
        self.seed = seed
        self.guidance_scale = guidance_scale
        # Core folders
        # self.models_dir = self.base_dir / "checkpoints"

        self.data_dir = self.base_dir / "data"
        self.tmp_dir = self.data_dir / "tmp"
        self.results_dir = self.data_dir / "results"
        self.backgrounds_dir = self.data_dir / "backgrounds"
        self.intermediate_videos_dir = self.data_dir / "intermediate_videos"
        self._ensure_dirs()

        # Specific file paths
        self.unet_config_path = self.base_dir / 'configs' / 'unet' / 'second_stage.yaml'

        self.inference_ckpt_path = self.models_dir / 'checkpoint-60000.pt'
        self.whisper_small_model_path = self.models_dir / "whisper" / "small.pt"
        self.whisper_tiny_model_path = self.models_dir / "whisper" / "tiny.pt"
        self.ref_video_path = self.data_dir / "ref_videos" / "ref_video_1min.mp4"
        self.saved_ref_video_data_path = self.data_dir / "ref_videos" / "ref_frames_data.pkl"
        self.silent_video_path = self.data_dir / "ref_videos" / "silent.mp4" 


    def _ensure_dirs(self):
        for d in [self.models_dir, self.data_dir, self.tmp_dir, self.results_dir, self.intermediate_videos_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def set_audio_path(self, path):
        self.audio_path = path

    def set_save_path(self, path):
        self.save_path = path

    def set_background_paths(self, backgrounds):
        self.background_paths = [os.path.join(self.backgrounds_dir, name + '.mp4') for name in backgrounds]

    def clean_dir(self, directory):
        """Remove all contents of a directory."""
        for path in os.listdir(directory):
            file = os.path.join(directory, path)
            os.remove(file)

    def __repr__(self):
        return f"<ProjectManager root={self.base_dir}>"
