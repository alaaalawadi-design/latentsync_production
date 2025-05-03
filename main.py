
from pathlib import Path
from blocks.lipsync_block import LibSyncBlock
from blocks.preprocessing_block import PreProcessingBlock
from blocks.postprocessing_block import PostProcessingBlock
from blocks.frame_interpolation_block import FrameIntrpolationBlock
from projectmanager import ProjectManager
from ref_video_manager import ReferenceVideoManager

class App:
    def __init__(self, 
                project_manager,
                ref_video_manager
                ):
        
        self.is_first_video = True
        self.fps = 25
        self.project_manager = project_manager
        self.ref_video_manager = ref_video_manager

        self.preprocessing_block = PreProcessingBlock() 
        self.lipsync_block = LibSyncBlock(self.project_manager.base_dir,
                                          self.project_manager.unet_config_path,
                                          self.project_manager.inference_ckpt_path,
                                          self.project_manager.whisper_small_model_path,
                                          self.project_manager.whisper_tiny_model_path,
                                          self.project_manager.ref_videos_dir, 
                                          self.ref_video_manager.ref_video_versions,
                                          self.project_manager.device, 
                                          self.project_manager.seed
                                          )
        self.frame_interpolation_block = FrameIntrpolationBlock(self.project_manager.models_dir)
        self.postprocessing_block = PostProcessingBlock()
        
        self.silent_frames = {}
        for version in self.ref_video_manager.ref_video_versions:
            silent_video_path = os.path.join(self.project_manager.ref_videos_dir, version, 'silent.mp4')
            first_frame, last_frame = self.preprocessing_block.get_first_last_frames(silent_video_path)
            self.silent_frames[version] = {
                'first': first_frame,
                'last': last_frame
            }
        
    def cleanup(self):
        for dir in [self.project_manager.results_dir, self.project_manager.intermediate_videos_dir, self.project_manager.tmp_dir]:
            self.project_manager.clean_dir(dir)
        
    def set_silent_frames(self, version):
        self.first_silent_frame = self.silent_frames[version]['first']
        self.last_silent_frame = self.silent_frames[version]['last']

    def run(self,
            ref_video_version 
            ):
        
        ref_video_data = ref_video_manager.set_ref_video(ref_video_version)
        self.lipsync_block.pipeline.prepare_ref_video_data(ref_video_data)
        self.set_silent_frames(ref_video_version)
        
        self.lipsync_block.execute(
            audio_path=self.project_manager.audio_path,
            video_out_path=self.project_manager.results_dir / "output.mp4",
            tmp_audio_path=self.project_manager.tmp_dir / "modified_audio.wav", 
            guidance_scale=self.project_manager.guidance_scale

        )

        if not self.is_first_video:
            first_frame = self.lipsync_block.get_first_gen_frame()
            intermediate_video1_path = self.project_manager.intermediate_videos_dir / "video1.mp4"
            self.frame_interpolation_block.execute(self.last_silent_frame, first_frame, self.fps, save_path=intermediate_video1_path)
        else:
            self.is_first_video = False
            intermediate_video1_path = None


        last_frame = self.lipsync_block.get_last_gen_frame()
        intermediate_video2_path = self.project_manager.intermediate_videos_dir / "video2.mp4"
        self.frame_interpolation_block.execute(last_frame, self.first_silent_frame, self.fps, save_path=intermediate_video2_path) 

        silent_video_path = os.path.join(self.project_manager.ref_videos_dir, ref_video_version, 'silent.mp4')
        if intermediate_video1_path is not None:
            videos = [intermediate_video1_path, self.project_manager.results_dir / "output.mp4", intermediate_video2_path, silent_video_path]
            speed_up_videos = [intermediate_video1_path, intermediate_video2_path]
        else:
            videos = [self.project_manager.results_dir / "output.mp4", intermediate_video2_path, silent_video_path]
            speed_up_videos = [intermediate_video2_path]

        self.postprocessing_block.execute(videos, speed_up_videos, output_file=self.project_manager.save_path)
        self.cleanup()


if __name__ == "__main__":
    
    base_dir=Path('.')
    models_dir = Path('/media/ehab/46EEC3E77E2602C6/Cyshield/LatentSync/checkpoints')
        
    project_manager = ProjectManager(base_dir, models_dir)
    ref_video_manager = ReferenceVideoManager(project_manager.ref_videos_dir)
    lipsync_app = App(project_manager, ref_video_manager)
    
    
    # import time 
    # s_time = time.time()
    audio_path = "/home/ehab/Downloads/audio2.wav"
    save_path = "/home/ehab/Downloads/out_sunset.mp4"
    ref_video_version = 'sunset'
    project_manager.set_audio_path(audio_path)
    project_manager.set_save_path(save_path)
    lipsync_app.run(ref_video_version)    
    # e_time = time.time()
    # print(e_time-s_time)
    
    
    
    audio_path = "/home/ehab/Downloads/out_7.wav"
    save_path = "/home/ehab/Downloads/out_morning.mp4"
    ref_video_version = 'morning'
    project_manager.set_audio_path(audio_path)
    project_manager.set_save_path(save_path)
    lipsync_app.run(ref_video_version)    
    

    audio_path = "/home/ehab/Downloads/audio2.wav"
    save_path = "/home/ehab/Downloads/out_night.mp4"
    ref_video_version = 'night'
    project_manager.set_audio_path(audio_path)
    project_manager.set_save_path(save_path)
    lipsync_app.run(ref_video_version) 

    audio_path = "/home/ehab/Downloads/out_7.wav"
    save_path = "/home/ehab/Downloads/out_noon.mp4"
    ref_video_version = 'noon'
    project_manager.set_audio_path(audio_path)
    project_manager.set_save_path(save_path)
    lipsync_app.run(ref_video_version) 

