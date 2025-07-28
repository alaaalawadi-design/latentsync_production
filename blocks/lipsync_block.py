from omegaconf import OmegaConf
import torch
import os 
import cv2
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from diffusers.utils.import_utils import is_xformers_available
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature
from latentsync.utils.image_processor import  ImageProcessor  , load_fixed_mask


class LibSyncBlock:
    def __init__(self, 
                base_dir,
                unet_config_path,
                inference_ckpt_path,
                whisper_small_model_path,
                whisper_tiny_model_path,
                detector_path,
                ref_video_path,
                saved_ref_video_data_path, 
                device='cuda',
                seed=None
                ):
    
        configs_dir = os.path.join(base_dir, "configs")
        scheduler = DDIMScheduler.from_pretrained(configs_dir)
        
        try:
            if not os.path.exists(unet_config_path):
                raise FileNotFoundError(f"Config file not found: {unet_config_path}")

            self.config = OmegaConf.load(str(unet_config_path))
        except Exception as e:
            print(f"Error loading config: {e}")
        
        if self.config.model.cross_attention_dim == 768:
            whisper_model_path = whisper_small_model_path
        elif self.config.model.cross_attention_dim == 384:
            whisper_model_path = whisper_tiny_model_path
        else:
            raise NotImplementedError("cross_attention_dim must be 768 or 384")
        audio_encoder = Audio2Feature(model_path=whisper_model_path, device=device, num_frames=self.config.data.num_frames)
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(self.config.model),
            inference_ckpt_path,  
            device="cpu",
        )
        unet = unet.to(dtype=torch.float16)
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()

        image_processor = ImageProcessor(
            detector_path=detector_path,
            resolution=256,
            device="cuda"
        )


        self.pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
            image_processor=image_processor
        ).to(device)
        if seed :
            set_seed(seed)
        else:    
            torch.seed()

        self.pipeline.prepare_ref_video(ref_video_path, saved_ref_video_data_path)

    def execute(self, 
            audio_path, 
            video_out_path, 
            tmp_audio_path,
            guidance_scale=1.0
            ):
        first_frame, last_frame = self.pipeline(
            audio_path=audio_path,
            tmp_audio_path=tmp_audio_path, 
            video_out_path=video_out_path,
            num_frames=self.config.data.num_frames,
            num_inference_steps=self.config.run.inference_steps,
            guidance_scale=guidance_scale,
            weight_dtype=torch.float16,
            width=self.config.data.resolution,
            height=self.config.data.resolution,
                # # Add default initialization - you'll need to set proper values


        )
        self.set_first_gen_frame(first_frame)
        self.set_last_gen_frame(last_frame)

    def set_first_gen_frame(self, first_frame):
        self.first_frame = first_frame

    def get_first_gen_frame(self):
        return cv2.cvtColor(self.first_frame, cv2.COLOR_RGB2BGR)

    def set_last_gen_frame(self, last_frame):
        self.last_frame = last_frame

    def get_last_gen_frame(self):
        return cv2.cvtColor(self.last_frame, cv2.COLOR_RGB2BGR)
        

        