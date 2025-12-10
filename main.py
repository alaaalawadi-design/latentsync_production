from pathlib import Path
from blocks.lipsync_block import LibSyncBlock
from blocks.preprocessing_block import PreProcessingBlock
from blocks.postprocessing_block import PostProcessingBlock
from blocks.frame_interpolation_block import FrameIntrpolationBlock
from projectmanager import ProjectManager

class App:
    def __init__(self, 
                project_manager,
                ):
        
        self.is_first_video = True
        self.fps = 25
        self.project_manager = project_manager

        self.preprocessing_block = PreProcessingBlock() 
        self.lipsync_block = LibSyncBlock(self.project_manager.base_dir,
                                          self.project_manager.unet_config_path,
                                          self.project_manager.inference_ckpt_path,
                                          self.project_manager.whisper_small_model_path,
                                          self.project_manager.whisper_tiny_model_path,
                                          self.project_manager.detector_path,
                                          self.project_manager.ref_video_path, 
                                          self.project_manager.saved_ref_video_data_path,
                                          self.project_manager.device, 
                                          self.project_manager.seed
                                          )
        self.frame_interpolation_block = FrameIntrpolationBlock(self.project_manager.models_dir)
        self.postprocessing_block = PostProcessingBlock()
        self.first_silent_frame, self.last_silent_frame = self.preprocessing_block.get_first_last_frames(self.project_manager.silent_video_path)
        
        
    def cleanup(self):
        for dir in [self.project_manager.results_dir, self.project_manager.intermediate_videos_dir, self.project_manager.tmp_dir]:
            self.project_manager.clean_dir(dir)
        
    def run(self, 
            ):

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


        if intermediate_video1_path is not None:
            videos = [intermediate_video1_path, self.project_manager.results_dir / "output.mp4", intermediate_video2_path, self.project_manager.silent_video_path]
            speed_up_videos = [intermediate_video1_path, intermediate_video2_path]
        else:
            videos = [self.project_manager.results_dir / "output.mp4", intermediate_video2_path, self.project_manager.silent_video_path]
            speed_up_videos = [intermediate_video2_path]

        self.postprocessing_block.execute(videos, speed_up_videos, self.project_manager.results_dir / "final_output.mp4", 
                                          self.project_manager.background_paths, self.project_manager.save_path)
        self.cleanup()




# if __name__ == "__main__":
    
    # base_dir=Path('.')
    # models_dir = Path('../checkpoints/')    
    # project_manager = ProjectManager(base_dir, models_dir)
    # lipsync_app = App(project_manager)
    
    
    # import time 
    # s_time = time.time()
    # audio_path = "/home/administrator/disk2/alaa/GeneAI2/latentsync_production/test_data/ict_waves/4.wav"
    # save_path = "/home/administrator/disk2/alaa/GeneAI2/latentsync_production/cysync/"
    # # backgrounds = ['morning', 'night', 'noon', 'sunset']
    # backgrounds = ['sunset']
    # project_manager.set_audio_path(audio_path)
    # project_manager.set_save_path(save_path)
    # project_manager.set_background_paths(backgrounds)
    # lipsync_app.run()    
    # e_time = time.time()
    # print( "all lipsync time with save ", e_time-s_time)