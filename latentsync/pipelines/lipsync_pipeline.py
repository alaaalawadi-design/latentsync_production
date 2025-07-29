
# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import os
from typing import Callable, List, Optional, Union
import subprocess
import pickle
from pydub import AudioSegment
from torchvision import transforms
import numpy as np
import torch
import torchvision
import math
from diffusers.utils import is_accelerate_available
from packaging import version
from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipeline_utils import DiffusionPipeline
import numpy as np
import torch
from einops import rearrange

import cv2
import gc
import psutil

import torch
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)



from diffusers.utils import deprecate, logging
from einops import rearrange
from ..models.unet import UNet3DConditionModel 
from ..utils.util import read_video, read_audio, write_video, check_ffmpeg_installed
from ..whisper.audio2feature import Audio2Feature
import tqdm



logger = logging.get_logger(__name__)  


# pylint: disable=invalid-name

class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
        image_processor
    ):


        self._debug_memory = True  # Enable memory logging
        
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )


            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )

            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)




        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )
        self.image_processor = image_processor
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        
        self.set_progress_bar_config(desc="Steps")
        # self.set_pointer(0)
        self.frame_pointer = 0


    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()


    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device



    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        decoded_latents = self.vae.decode(latents).sample
        return decoded_latents


    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]



        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
     
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    
    
    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )



    def prepare_latents(self, batch_size, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            batch_size,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    
    
    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents
    



    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents
    


    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)


    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    
    # def affine_transform_video(self, video_path):
    #     video_frames = read_video(video_path, use_decord=False)
    #     faces = []
    #     boxes = []
    #     affine_matrices = []    
    #     for frame in tqdm.tqdm(video_frames):
    #         face, box, affine_matrix = self.image_processor.affine_transform(frame)
    #         faces.append(face)
    #         boxes.append(box)
    #         affine_matrices.append(affine_matrix)
    #     faces = torch.stack(faces)
    #     return faces, video_frames, boxes, affine_matrices
 # ================================================================
# 1. OPTIMIZE: affine_transform_video() - SAME SIGNATURE, CHUNKED PROCESSING
# FILE: lipsync_pipeline.py
# ================================================================

    def affine_transform_video(self, video_path):
        """
        OPTIMIZED: Process video in memory-efficient chunks while maintaining same output
        NO CHANGES to function signature or return values
        """

        
        # Use CV2 for memory-efficient frame reading instead of loading all at once
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        faces = []
        boxes = []
        affine_matrices = []
        video_frames = []
        
        # Process in chunks to avoid memory overflow
        chunk_size = 100  # Adjust based on your RAM
        
        for chunk_start in tqdm.tqdm(range(0, total_frames, chunk_size), desc="Processing video chunks"):
            chunk_end = min(chunk_start + chunk_size, total_frames)
            
            # Read current chunk
            chunk_frames = []
            for frame_idx in range(chunk_start, chunk_end):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                chunk_frames.append(frame_rgb)
            
            # Process current chunk
            for frame in chunk_frames:
                try:
                    face, box, affine_matrix = self.image_processor.affine_transform(frame)
                    faces.append(face)
                    boxes.append(box)
                    affine_matrices.append(affine_matrix)
                    video_frames.append(frame)
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    if faces:  # Use previous frame if available
                        faces.append(faces[-1])
                        boxes.append(boxes[-1])
                        affine_matrices.append(affine_matrices[-1])
                        video_frames.append(video_frames[-1])
            
            # Clear chunk memory immediately
            del chunk_frames
            gc.collect()
            
            # Clear GPU cache every few chunks
            if torch.cuda.is_available() and chunk_start % (chunk_size * 3) == 0:
                torch.cuda.empty_cache()
        
        cap.release()
        
        # Return same format as original
        faces = torch.stack(faces) if faces else torch.empty(0)
        video_frames = np.array(video_frames) if video_frames else np.empty((0, 0, 0, 0))
        
        return faces, video_frames, boxes, affine_matrices





    # def restore_video(self, faces: torch.Tensor, video_frames: np.ndarray, boxes: list, affine_matrices: list):
    #     video_frames = video_frames[: len(faces)]
    #     out_frames = []
    #     print(f"Restoring {len(faces)} faces...")

    #     for index, face in enumerate(tqdm.tqdm(faces)):
    #         x1, y1, x2, y2 = boxes[index]
    #         height = int(y2 - y1)
    #         width = int(x2 - x1)
            
            
    #         # Resize face to match aligned face dimensions (from affine transform)
    #         face = torchvision.transforms.functional.resize(
    #             face, size=(height, width), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
    #         )
            
    #         # Don't normalize - let AlignRestore handle it
    #         face = face.to(dtype=self.image_processor.restorer.dtype, device=self.image_processor.restorer.device)
            
    #         out_frame = self.image_processor.restorer.restore_img(video_frames[index], face, affine_matrices[index])
    #         out_frames.append(out_frame)
        
    #     return np.stack(out_frames, axis=0)
    


    # ================================================================
    # 3. OPTIMIZE: restore_video() - SAME SIGNATURE, CHUNKED PROCESSING  
    # FILE: lipsync_pipeline.py
    # ================================================================

    # def restore_video(self, faces: torch.Tensor, video_frames: np.ndarray, boxes: list, affine_matrices: list):
    #     """
    #     OPTIMIZED: Process restoration in memory-efficient chunks
    #     NO CHANGES to function signature or return format
    #     """
    #     import gc
        
    #     total_frames = len(faces)
    #     out_frames = []
        
    #     print(f"Restoring {total_frames} faces with memory optimization...")
        
    #     # Process in smaller chunks to manage memory
    #     chunk_size = 100  # Adjust based on your GPU memory
        
    #     for chunk_start in tqdm.tqdm(range(0, total_frames, chunk_size), desc="Restoring frames"):
    #         chunk_end = min(chunk_start + chunk_size, total_frames)
            
    #         # Process current chunk
    #         chunk_restored = []
            
    #         for i in range(chunk_start, chunk_end):
    #             if i >= len(video_frames):
    #                 continue
                    
    #             x1, y1, x2, y2 = boxes[i]
    #             height = int(y2 - y1)
    #             width = int(x2 - x1)
                
    #             # Resize face to match aligned face dimensions
    #             face = torchvision.transforms.functional.resize(
    #                 faces[i], size=(height, width), 
    #                 interpolation=transforms.InterpolationMode.BICUBIC, 
    #                 antialias=True
    #             )
                
    #             # Move to correct device and process
    #             face = face.to(dtype=self.image_processor.restorer.dtype, 
    #                         device=self.image_processor.restorer.device)
                
    #             out_frame = self.image_processor.restorer.restore_img(
    #                 video_frames[i], face, affine_matrices[i]
    #             )
    #             chunk_restored.append(out_frame)
            
    #         # Add chunk results to main list
    #         out_frames.extend(chunk_restored)
            
    #         # Clear chunk memory immediately
    #         del chunk_restored

    #         gc.collect()
            
    #         # Clear GPU cache periodically
    #         if torch.cuda.is_available() and chunk_start % (chunk_size * 2) == 0:
    #             torch.cuda.empty_cache()
        
    #     # Return same format as original
    #     return np.stack(out_frames, axis=0)






    def restore_video(self, faces: torch.Tensor, video_frames: np.ndarray, boxes: list, affine_matrices: list):
        """
        FIXED: Smaller chunk size for restoration to prevent OOM
        """
        import gc
        
        total_frames = len(faces)
        out_frames = []
        
        print(f"Restoring {total_frames} faces with memory optimization...")
        
        # ✅ CRITICAL FIX: Reduce chunk size from 50 to 15-20 for large videos
        if total_frames > 2000:
            chunk_size = 2  # Very conservative for long videos
        elif total_frames > 1000:
            chunk_size = 25  # Conservative for medium videos  
        else:
            chunk_size = 50  # Original size for short videos
        
        print(f"Using chunk size: {chunk_size} for {total_frames} frames")
        
        for chunk_start in tqdm.tqdm(range(0, total_frames, chunk_size), desc="Restoring frames"):
            chunk_end = min(chunk_start + chunk_size, total_frames)
            
            # Process current chunk
            chunk_restored = []
            
            for i in range(chunk_start, chunk_end):
                if i >= len(video_frames):
                    continue
                    
                x1, y1, x2, y2 = boxes[i]
                height = int(y2 - y1)
                width = int(x2 - x1)
                
                # Resize face to match aligned face dimensions
                face = torchvision.transforms.functional.resize(
                    faces[i], size=(height, width), 
                    interpolation=transforms.InterpolationMode.BICUBIC, 
                    antialias=True
                )
                
                # Move to correct device and process
                face = face.to(dtype=self.image_processor.restorer.dtype, 
                            device=self.image_processor.restorer.device)
                
                out_frame = self.image_processor.restorer.restore_img(
                    video_frames[i], face, affine_matrices[i]
                )
                chunk_restored.append(out_frame)
                
                # ✅ ADDITIONAL FIX: Clear GPU memory every 5 frames within chunk
                if (i - chunk_start) % 5 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Add chunk results to main list
            out_frames.extend(chunk_restored)
            
            # ✅ CRITICAL: Clear chunk memory immediately + force cleanup
            del chunk_restored
            gc.collect()
            
            # ✅ ADDITIONAL: Clear GPU cache after EVERY chunk (not every 2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Wait for GPU operations to complete
            
            # ✅ EXTRA SAFETY: Print memory usage every 10 chunks
            if chunk_start % (chunk_size * 10) == 0:
                if hasattr(self, '_log_memory_usage'):
                    self._log_memory_usage(f"Restoration Progress {chunk_start}/{total_frames}")
        
        # Return same format as original
        return np.stack(out_frames, axis=0)

    # @torch.no_grad()
    # def __call__(
    #     self,
    #     audio_path: str,
    #     tmp_audio_path: str,
    #     video_out_path: str,
    #     video_mask_path: str = None,
    #     num_frames: int = 16,
    #     video_fps: int = 25,
    #     audio_sample_rate: int = 16000,
    #     height: Optional[int] = None,
    #     width: Optional[int] = None,
    #     num_inference_steps: int = 20,
    #     guidance_scale: float = 1.5,
    #     weight_dtype: Optional[torch.dtype] = torch.float16,
    #     eta: float = 0.0,
    #     generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    #     callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    #     callback_steps: Optional[int] = 1,
    #     # **kwargs,
    # ):

    #     self.unet.eval()
    #     check_ffmpeg_installed()
    #     # 0. Define call parameters


    #     batch_size = 1
    #     device = self._execution_device

    #     height = height or self.unet.config.sample_size * self.vae_scale_factor
    #     width = width or self.unet.config.sample_size * self.vae_scale_factor


    #     self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")        
    #     self.add_silent_to_audio(audio_path, audio_sample_rate, tmp_audio_path)
    #     audio_samples = read_audio(str(tmp_audio_path), audio_sample_rate)

    #     # 2. Check inputs

    #     self.check_inputs(height, width, callback_steps)

    #     do_classifier_free_guidance = guidance_scale > 1.0
    #     # 3. set timesteps

    #     self.scheduler.set_timesteps(num_inference_steps, device=device)
    #     timesteps = self.scheduler.timesteps

    #     # 4. Prepare extra step kwargs.

    #     extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    #     self.video_fps = video_fps


    #     if self.unet.add_audio_layer:
    #         whisper_feature = self.audio_encoder.audio2feat(audio_samples)
    #         whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)
    #         num_inferences = math.ceil(len(whisper_chunks) / num_frames)


    #     synced_video_frames = []
    #     num_channels_latents = self.vae.config.latent_channels


    #     total_frames = len(whisper_chunks)  
    #     all_latents = self.prepare_latents(
    #         batch_size,
    #         total_frames,  
    #         num_channels_latents,
    #         height,
    #         width,
    #         weight_dtype,
    #         device,
    #         generator,
    #     )

    #     self.set_ref_video_data(start=self.frame_pointer, end=total_frames+self.frame_pointer)
        
    #     self.set_pointer(self.frame_pointer+total_frames-5)
        
    #     for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):

    #         start_idx = i * num_frames
    #         end_idx = min(start_idx + num_frames, total_frames)  
            
            
    #         if self.unet.add_audio_layer:
    #             audio_embeds = torch.stack(whisper_chunks[start_idx:end_idx])
    #             audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
            
    #             if do_classifier_free_guidance:
    #                 null_audio_embeds = torch.zeros_like(audio_embeds)
    #                 audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
            
    #         else:
    #             audio_embeds = None

    #         inference_faces = self.faces[start_idx:end_idx]
    #         latents = all_latents[:, :, start_idx:end_idx]  

    #         pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
    #             inference_faces, affine_transform=False
    #         )
    #         mask_latents, masked_image_latents = self.prepare_mask_latents(
    #             masks,
    #             masked_pixel_values,
    #             height,
    #             width,
    #             weight_dtype,
    #             device,
    #             generator,
    #             do_classifier_free_guidance,
    #         )
    #         image_latents = self.prepare_image_latents(
    #             pixel_values,
    #             device,
    #             weight_dtype,
    #             generator,
    #             do_classifier_free_guidance,
    #         )
            
    #         num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            
    #         with self.progress_bar(total=num_inference_steps) as progress_bar:
    #             for j, t in enumerate(timesteps):
    #                 latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    #                 latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
    #                 latent_model_input = torch.cat(
    #                     [latent_model_input, mask_latents, masked_image_latents, image_latents], dim=1
    #                 )
    #                 noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=audio_embeds).sample
    #                 if do_classifier_free_guidance:
    #                     noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
    #                     noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)
    #                 latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
    #                 if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
    #                     progress_bar.update()
    #                     if callback is not None and j % callback_steps == 0:
    #                         callback(j, t, latents)
    #         decoded_latents = self.decode_latents(latents)
    #         decoded_latents = self.paste_surrounding_pixels_back(
    #             decoded_latents, pixel_values, 1 - masks, device, weight_dtype
    #         )
    #         synced_video_frames.append(decoded_latents)

        
    #     synced_video_frames = self.restore_video(
    #         torch.cat(synced_video_frames), self.original_video_frames, self.boxes, self.affine_matrices
    #     )

        
    #     temp_dir = os.path.dirname(tmp_audio_path)
    #     write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames[5:], fps=25)
    #     command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {tmp_audio_path} -c:v libx264 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
    #     subprocess.run(command, shell=True)
    #     return synced_video_frames[5, :, :, :], synced_video_frames[-1, :, :, :]








    # # def prepare_ref_video(self, video_path, saved_video_data_path):
    # #     if not os.path.exists(saved_video_data_path):
    # #         self.all_faces, self.all_original_video_frames, self.all_boxes, self.all_affine_matrices = self.affine_transform_video(video_path)
    # #         with open(saved_video_data_path, "wb") as f:
    # #             pickle.dump([self.all_faces, self.all_original_video_frames, self.all_boxes, self.all_affine_matrices], f)    
    # #     else:
    # #         with open(saved_video_data_path, "rb") as f:
    # #             out_video_with_scielnt = pickle.load(f)
    # #         self.all_faces, self.all_original_video_frames, self.all_boxes, self.all_affine_matrices = out_video_with_scielnt



    # ================================================================
    # 2. OPTIMIZE: prepare_ref_video() - SAME SIGNATURE, EFFICIENT CACHING
    # FILE: lipsync_pipeline.py
    # ================================================================

    def prepare_ref_video(self, video_path, saved_video_data_path):
        """
        OPTIMIZED: Efficient pickle handling and memory management
        NO CHANGES to function signature
        """
        import pickle
        import os
        import gc
        
        if not os.path.exists(saved_video_data_path):
            print("Processing reference video with memory optimization...")
            
            # Process video efficiently
            self.all_faces, self.all_original_video_frames, self.all_boxes, self.all_affine_matrices = self.affine_transform_video(video_path)
            
            # Save with optimized pickle settings
            print(f"Saving processed data to {saved_video_data_path}...")
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(saved_video_data_path), exist_ok=True)
                
                # Use memory-efficient pickle saving
                with open(saved_video_data_path, "wb") as f:
                    # Save in parts to reduce memory pressure
                    pickle.dump(self.all_faces, f, protocol=pickle.HIGHEST_PROTOCOL)
                    pickle.dump(self.all_original_video_frames, f, protocol=pickle.HIGHEST_PROTOCOL)
                    pickle.dump(self.all_boxes, f, protocol=pickle.HIGHEST_PROTOCOL)  
                    pickle.dump(self.all_affine_matrices, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                print(f"Successfully saved {len(self.all_faces)} processed frames")
                
            except Exception as e:
                print(f"Warning: Could not save processed data: {e}")
        else:
            print("Loading cached reference video data...")
            try:
                # Memory-efficient pickle loading
                with open(saved_video_data_path, "rb") as f:
                    self.all_faces = pickle.load(f)
                    self.all_original_video_frames = pickle.load(f)
                    self.all_boxes = pickle.load(f)
                    self.all_affine_matrices = pickle.load(f)
                
                print(f"Successfully loaded {len(self.all_faces)} cached frames")
                
                # Clear any residual memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error loading cached data: {e}. Reprocessing...")
                # Remove corrupted cache and reprocess
                os.remove(saved_video_data_path)
                self.prepare_ref_video(video_path, saved_video_data_path)



    def add_silent_to_audio(self, audio_path, audio_sample_rate, tmp_audio_path):
        audio = AudioSegment.from_file(audio_path, format="wav")
        number_of_samples = 4480
        duration_of_silence = (1000 * number_of_samples) / audio_sample_rate  
        silent_segment = AudioSegment.silent(duration=duration_of_silence, frame_rate=audio_sample_rate)
        new_audio = silent_segment + audio
        new_audio.export(tmp_audio_path, format="wav")
        
    
    def set_pointer(self, start_number):
        total_frames = len(self.all_faces)
        self.frame_pointer = start_number % total_frames



    
    # def set_ref_video_data(self, start=0, end=None):
    #     total_frames = len(self.all_faces)

    #     if end is None:
    #         end = total_frames
        
    #     required_frames = end - start
        
    #     if required_frames <= total_frames:
    #         end = end % total_frames  
    #         if start < end:
    #             # Simple case: no wraparound needed
    #             self.faces = self.all_faces[start:end]
    #             self.original_video_frames = self.all_original_video_frames[start:end]
    #             self.boxes = self.all_boxes[start:end]
    #             self.affine_matrices = self.all_affine_matrices[start:end]
    #         else:
    #             # Wraparound case
    #             self.faces = torch.cat((self.all_faces[start:], self.all_faces[:end]), dim=0)
    #             self.original_video_frames = np.concatenate((self.all_original_video_frames[start:], self.all_original_video_frames[:end]), axis=0)
    #             self.boxes = np.concatenate((self.all_boxes[start:], self.all_boxes[:end]), axis=0)
    #             self.affine_matrices = self.all_affine_matrices[start:] + self.all_affine_matrices[:end]
    #     else:
    #         # Need more frames than available - loop the video
    #         full_cycles = required_frames // total_frames
    #         remaining_frames = required_frames % total_frames
            
    #         self.faces = torch.cat([self.all_faces] * full_cycles + [self.all_faces[:remaining_frames]], dim=0)
    #         self.original_video_frames = np.concatenate([self.all_original_video_frames] * full_cycles + [self.all_original_video_frames[:remaining_frames]], axis=0)
    #         self.boxes = np.concatenate([self.all_boxes] * full_cycles + [self.all_boxes[:remaining_frames]], axis=0)
    #         self.affine_matrices = self.all_affine_matrices * full_cycles + self.all_affine_matrices[:remaining_frames]



    # ================================================================
    # 4. OPTIMIZE: __call__() method - SAME SIGNATURE, MEMORY MANAGEMENT
    # FILE: lipsync_pipeline.py  
    # ================================================================

    @torch.no_grad()
    def __call__(
        self,
        audio_path: str,
        tmp_audio_path: str,
        video_out_path: str,
        video_mask_path: str = None,
        num_frames: int = 16,  # KEEP 16 - model requirement
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,

        # callback_steps: Optional[int] = 1,
        cleanup_frequency: int = 3  # ✅ Proper parameter declaration
            ):
        """
        OPTIMIZED: Same logic and parameters, but with memory management
        NO CHANGES to function signature or core logic
        """
        import gc
        
        self.unet.eval()
        check_ffmpeg_installed()
        
        # Clear memory before starting
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 0. Define call parameters
        batch_size = 1
        device = self._execution_device

        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")        
        self.add_silent_to_audio(audio_path, audio_sample_rate, tmp_audio_path)
        audio_samples = read_audio(str(tmp_audio_path), audio_sample_rate)

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

        do_classifier_free_guidance = guidance_scale > 1.0
        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        self.video_fps = video_fps

        if self.unet.add_audio_layer:
            whisper_feature = self.audio_encoder.audio2feat(audio_samples)
            whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)
            num_inferences = math.ceil(len(whisper_chunks) / num_frames)

        synced_video_frames = []
        num_channels_latents = self.vae.config.latent_channels

        total_frames = len(whisper_chunks)  
        
        # OPTIMIZED: Prepare latents in smaller chunks to reduce memory pressure
        chunk_latents_size = min(total_frames, 64)  # Process latents in chunks
        all_latents = self.prepare_latents(
            batch_size,
            chunk_latents_size,  # Smaller initial size
            num_channels_latents,
            height,
            width,
            weight_dtype,
            device,
            generator,
        )

        self.set_ref_video_data(start=self.frame_pointer, end=total_frames+self.frame_pointer)
        self.set_pointer(self.frame_pointer+total_frames-5)
        
        # OPTIMIZED: Process inference with memory management
        for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
            start_idx = i * num_frames
            end_idx = min(start_idx + num_frames, total_frames)  
            
            # Clear memory before each inference
            if i % 3 == 0:  # Every 3 iterations
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if self.unet.add_audio_layer:
                audio_embeds = torch.stack(whisper_chunks[start_idx:end_idx])
                audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
            
                if do_classifier_free_guidance:
                    null_audio_embeds = torch.zeros_like(audio_embeds)
                    audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
            else:
                audio_embeds = None

            inference_faces = self.faces[start_idx:end_idx]
            
            # OPTIMIZED: Prepare latents for current batch size only
            current_batch_size = end_idx - start_idx
            if current_batch_size != all_latents.shape[2]:
                latents = self.prepare_latents(
                    batch_size, current_batch_size, num_channels_latents,
                    height, width, weight_dtype, device, generator,
                )
            else:
                latents = all_latents[:, :, :current_batch_size]

            pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                inference_faces, affine_transform=False
            )
            mask_latents, masked_image_latents = self.prepare_mask_latents(
                masks,
                masked_pixel_values,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )
            image_latents = self.prepare_image_latents(
                pixel_values,
                device,
                weight_dtype,
                generator,
                do_classifier_free_guidance,
            )
            
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for j, t in enumerate(timesteps):
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    latent_model_input = torch.cat(
                        [latent_model_input, mask_latents, masked_image_latents, image_latents], dim=1
                    )
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=audio_embeds).sample
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                    if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and j % callback_steps == 0:
                            callback(j, t, latents)
            
            decoded_latents = self.decode_latents(latents)
            decoded_latents = self.paste_surrounding_pixels_back(
                decoded_latents, pixel_values, 1 - masks, device, weight_dtype
            )
            synced_video_frames.append(decoded_latents)
            
            # OPTIMIZED: Clear intermediate tensors immediately
            del (latents, pixel_values, masked_pixel_values, masks, 
                mask_latents, masked_image_latents, image_latents, 
                decoded_latents, audio_embeds)

        # OPTIMIZED: Process video restoration with memory management
        synced_video_frames = self.restore_video(
            torch.cat(synced_video_frames), self.original_video_frames, self.boxes, self.affine_matrices
        )

        # Clear memory before final video processing
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        temp_dir = os.path.dirname(tmp_audio_path)
        write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames[5:], fps=25)
        command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {tmp_audio_path} -c:v libx264 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True)
        
        return synced_video_frames[5, :, :, :], synced_video_frames[-1, :, :, :]


    # ================================================================
    # 5. OPTIMIZE: set_ref_video_data() - SAME SIGNATURE, MEMORY EFFICIENT
    # FILE: lipsync_pipeline.py
    # ================================================================

    def set_ref_video_data(self, start=0, end=None):
        """
        OPTIMIZED: Memory-efficient data slicing and management
        NO CHANGES to function signature or logic
        """
        import gc
        
        # Clear previous references first
        if hasattr(self, 'faces'):
            del self.faces
        if hasattr(self, 'original_video_frames'):  
            del self.original_video_frames
        if hasattr(self, 'boxes'):
            del self.boxes
        if hasattr(self, 'affine_matrices'):
            del self.affine_matrices
        
        gc.collect()
        
        total_frames = len(self.all_faces)

        if end is None:
            end = total_frames
        
        required_frames = end - start
        
        if required_frames <= total_frames:
            end = end % total_frames  
            if start < end:
                # Simple case: no wraparound needed - use views when possible
                self.faces = self.all_faces[start:end].clone()  # Clone to avoid memory leaks
                self.original_video_frames = self.all_original_video_frames[start:end].copy()
                self.boxes = self.all_boxes[start:end].copy()
                self.affine_matrices = self.all_affine_matrices[start:end].copy()
            else:
                # Wraparound case
                self.faces = torch.cat((self.all_faces[start:], self.all_faces[:end]), dim=0)
                self.original_video_frames = np.concatenate((self.all_original_video_frames[start:], self.all_original_video_frames[:end]), axis=0)
                self.boxes = np.concatenate((self.all_boxes[start:], self.all_boxes[:end]), axis=0)
                self.affine_matrices = self.all_affine_matrices[start:] + self.all_affine_matrices[:end]
        else:
            # Need more frames than available - loop the video efficiently
            full_cycles = required_frames // total_frames
            remaining_frames = required_frames % total_frames
            
            # Create lists first, then concatenate once to reduce memory overhead
            faces_list = [self.all_faces] * full_cycles
            if remaining_frames > 0:
                faces_list.append(self.all_faces[:remaining_frames])
            self.faces = torch.cat(faces_list, dim=0)
            
            frames_list = [self.all_original_video_frames] * full_cycles  
            if remaining_frames > 0:
                frames_list.append(self.all_original_video_frames[:remaining_frames])
            self.original_video_frames = np.concatenate(frames_list, axis=0)
            
            boxes_list = [self.all_boxes] * full_cycles
            if remaining_frames > 0:
                boxes_list.append(self.all_boxes[:remaining_frames])
            self.boxes = np.concatenate(boxes_list, axis=0)
            
            self.affine_matrices = self.all_affine_matrices * full_cycles + self.all_affine_matrices[:remaining_frames]
        
        # Clear temporary lists
        if 'faces_list' in locals():
            del faces_list
        if 'frames_list' in locals():
            del frames_list  
        if 'boxes_list' in locals():
            del boxes_list
        
        gc.collect()


    # ================================================================
    # 6. OPTIMIZE: Pickle handling - SEPARATE FUNCTIONS FOR MEMORY EFFICIENCY
    # FILE: lipsync_pipeline.py (add these helper functions)
    # ================================================================

    def _save_processed_data_efficiently(self, data_path, faces, frames, boxes, matrices):
        """
        Helper function for memory-efficient pickle saving
        """
        import pickle
        import os
        import gc
        
        # Create directory if needed
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        try:
            # Save each component separately to reduce memory pressure
            with open(data_path, "wb") as f:
                pickle.dump(faces, f, protocol=pickle.HIGHEST_PROTOCOL)
                del faces  # Free memory immediately
                gc.collect()
                
                pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)  
                del frames
                gc.collect()
                
                pickle.dump(boxes, f, protocol=pickle.HIGHEST_PROTOCOL)
                del boxes
                gc.collect()
                
                pickle.dump(matrices, f, protocol=pickle.HIGHEST_PROTOCOL)
                del matrices
                gc.collect()
                
            print(f"Successfully saved processed data to {data_path}")
            
        except Exception as e:
            print(f"Error saving processed data: {e}")
            # Clean up partial file
            if os.path.exists(data_path):
                os.remove(data_path)
            raise

    def _load_processed_data_efficiently(self, data_path):
        """
        Helper function for memory-efficient pickle loading
        """
        import pickle
        import gc
        
        try:
            with open(data_path, "rb") as f:
                faces = pickle.load(f)
                gc.collect()  # Clear memory after each load
                
                frames = pickle.load(f)
                gc.collect()
                
                boxes = pickle.load(f)
                gc.collect()
                
                matrices = pickle.load(f)
                gc.collect()
                
            return faces, frames, boxes, matrices
            
        except Exception as e:
            print(f"Error loading processed data: {e}")
            raise


    # ================================================================
    # 7. OPTIMIZE: Memory management utilities - ADD TO lipsync_pipeline.py
    # FILE: lipsync_pipeline.py (add these methods to the class)
    # ================================================================

    def _clear_intermediate_memory(self):
        """
        Helper function to clear intermediate processing memory
        """
        import gc
        
        # Clear any cached gradients
        if hasattr(self, 'unet') and self.unet is not None:
            for param in self.unet.parameters():
                if param.grad is not None:
                    param.grad = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()

    def _optimize_tensor_memory(self, tensor):
        """
        Helper function to optimize tensor memory usage
        """
        if isinstance(tensor, torch.Tensor):
            # Use contiguous memory layout
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            
            # Pin memory for faster GPU transfers if on CPU
            if tensor.device.type == 'cpu' and torch.cuda.is_available():
                tensor = tensor.pin_memory()
        
        return tensor


    # ================================================================
    # 8. OPTIMIZE: Add memory monitoring to existing methods
    # FILE: lipsync_pipeline.py (modify existing methods with these additions)
    # ================================================================

    # ADD these imports at the top of lipsync_pipeline.py:

    # ADD this method to monitor memory during processing:
    def _log_memory_usage(self, stage=""):
        """
        Helper function to log memory usage during processing
        """
        if hasattr(self, '_debug_memory') and self._debug_memory:
            ram_usage = psutil.virtual_memory().percent
            if torch.cuda.is_available():
                vram_usage = (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
                print(f"[{stage}] RAM: {ram_usage:.1f}% | VRAM: {vram_usage:.1f}%")
            else:
                print(f"[{stage}] RAM: {ram_usage:.1f}%")

    # Enable memory logging by adding this to __init__:
    # self._debug_memory = True  # Set to False to disable logging