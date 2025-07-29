from pathlib import Path
from blocks.lipsync_block import LibSyncBlock
from blocks.preprocessing_block import PreProcessingBlock
from blocks.postprocessing_block import PostProcessingBlock
from blocks.frame_interpolation_block import FrameIntrpolationBlock
from projectmanager import ProjectManager


# Memory monitoring functions - no changes to existing classes needed
import psutil
import torch
import gc
import functools

def get_memory_usage():
    """Get current RAM and VRAM usage"""
    # RAM usage
    ram_info = psutil.virtual_memory()
    ram_used_gb = ram_info.used / (1024**3)
    ram_total_gb = ram_info.total / (1024**3)
    ram_percent = ram_info.percent
    
    # VRAM usage
    vram_used_gb = 0
    vram_total_gb = 0
    vram_percent = 0
    vram_reserved_gb = 0
    
    if torch.cuda.is_available():
        vram_used_bytes = torch.cuda.memory_allocated()
        vram_reserved_bytes = torch.cuda.memory_reserved()
        vram_used_gb = vram_used_bytes / (1024**3)
        vram_reserved_gb = vram_reserved_bytes / (1024**3)
        
        # Get total GPU memory
        gpu_props = torch.cuda.get_device_properties(0)
        vram_total_gb = gpu_props.total_memory / (1024**3)
        vram_percent = (vram_used_gb / vram_total_gb) * 100 if vram_total_gb > 0 else 0
    
    return {
        'ram_used_gb': ram_used_gb,
        'ram_total_gb': ram_total_gb,
        'ram_percent': ram_percent,
        'vram_used_gb': vram_used_gb,
        'vram_reserved_gb': vram_reserved_gb,
        'vram_total_gb': vram_total_gb,
        'vram_percent': vram_percent
    }

def print_memory_usage(stage=""):
    """Print current memory usage with optional stage label"""
    mem_info = get_memory_usage()
    print(f"\n{'='*60}")
    if stage:
        print(f"MEMORY USAGE - {stage}")
    else:
        print("MEMORY USAGE")
    print(f"{'='*60}")
    print(f"RAM:  {mem_info['ram_used_gb']:.2f} GB / {mem_info['ram_total_gb']:.2f} GB ({mem_info['ram_percent']:.1f}%)")
    if torch.cuda.is_available():
        print(f"VRAM: {mem_info['vram_used_gb']:.2f} GB / {mem_info['vram_total_gb']:.2f} GB ({mem_info['vram_percent']:.1f}%)")
        print(f"VRAM Reserved: {mem_info['vram_reserved_gb']:.2f} GB")
    else:
        print("VRAM: CUDA not available")
    print(f"{'='*60}\n")

def clear_gpu_cache():
    """Clear GPU cache and force garbage collection"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("🧹 Cleared GPU cache and ran garbage collection")

def quick_memory_check(label=""):
    """Quick one-line memory usage print"""
    ram = psutil.virtual_memory()
    ram_gb = ram.used / (1024**3)
    ram_percent = ram.percent
    
    if torch.cuda.is_available():
        vram_gb = torch.cuda.memory_allocated() / (1024**3)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        vram_percent = (vram_gb / total_vram_gb) * 100
        
        print(f"📊 [{label}] RAM: {ram_gb:.1f}GB ({ram_percent:.1f}%) | VRAM: {vram_gb:.1f}GB ({vram_percent:.1f}%)")
    else:
        print(f"📊 [{label}] RAM: {ram_gb:.1f}GB ({ram_percent:.1f}%) | VRAM: N/A")

def memory_diff(func):
    """Decorator to show memory difference before and after function execution"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Memory before
        mem_before = get_memory_usage()
        print(f"\n🔍 MEMORY MONITOR: {func.__name__} START")
        print(f"   RAM: {mem_before['ram_used_gb']:.2f} GB | VRAM: {mem_before['vram_used_gb']:.2f} GB")
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Memory after
        mem_after = get_memory_usage()
        ram_delta = mem_after['ram_used_gb'] - mem_before['ram_used_gb']
        vram_delta = mem_after['vram_used_gb'] - mem_before['vram_used_gb']
        
        print(f"   RAM: {mem_after['ram_used_gb']:.2f} GB (Δ{ram_delta:+.2f} GB)")
        print(f"   VRAM: {mem_after['vram_used_gb']:.2f} GB (Δ{vram_delta:+.2f} GB)")
        print(f"✅ MEMORY MONITOR: {func.__name__} END\n")
        
        return result
    return wrapper

def check_memory_warning(ram_threshold=85, vram_threshold=90):
    """Check if memory usage is getting high and warn user"""
    mem_info = get_memory_usage()
    warnings = []
    
    if mem_info['ram_percent'] > ram_threshold:
        warnings.append(f"⚠️  HIGH RAM USAGE: {mem_info['ram_percent']:.1f}%")
    
    if torch.cuda.is_available() and mem_info['vram_percent'] > vram_threshold:
        warnings.append(f"⚠️  HIGH VRAM USAGE: {mem_info['vram_percent']:.1f}%")
    
    if warnings:
        print("\n" + "🚨" * 20)
        for warning in warnings:
            print(warning)
        print("Consider reducing batch size or clearing cache!")
        print("🚨" * 20 + "\n")
        return True
    return False

def print_system_info():
    """Print system information at startup"""
    print(f"\n{'='*60}")
    print("🖥️  SYSTEM INFORMATION")
    print(f"{'='*60}")
    
    # CPU info
    print(f"CPU Count: {psutil.cpu_count()} cores")
    print(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")
    
    # RAM info
    ram = psutil.virtual_memory()
    print(f"Total RAM: {ram.total / (1024**3):.2f} GB")
    
    # GPU info
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU Count: {gpu_count}")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Total VRAM: {props.total_memory / (1024**3):.2f} GB")
    else:
        print("CUDA not available")
    
    print(f"{'='*60}\n")

# Global variables to track peak usage
_peak_ram = 0
_peak_vram = 0

def update_peak_memory():
    """Update peak memory tracking"""
    global _peak_ram, _peak_vram
    mem_info = get_memory_usage()
    _peak_ram = max(_peak_ram, mem_info['ram_used_gb'])
    _peak_vram = max(_peak_vram, mem_info['vram_used_gb'])

def print_peak_memory():
    """Print peak memory usage during session"""
    global _peak_ram, _peak_vram
    print(f"\n{'='*60}")
    print("🏆 PEAK MEMORY USAGE DURING SESSION")
    print(f"{'='*60}")
    print(f"Peak RAM:  {_peak_ram:.2f} GB")
    print(f"Peak VRAM: {_peak_vram:.2f} GB")
    print(f"{'='*60}\n")

def reset_peak_memory():
    """Reset peak memory tracking"""
    global _peak_ram, _peak_vram
    _peak_ram = 0
    _peak_vram = 0



class App:
    def __init__(self, 
                project_manager,
                ):
        
        self.is_first_video = True
        self.fps = 25
        self.project_manager = project_manager



        print_memory_usage("INITIALIZATION START")


        self.preprocessing_block = PreProcessingBlock() 


        quick_memory_check("After PreProcessing Init")


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
        

        print_memory_usage("AFTER LIPSYNC BLOCK INIT")
        self.frame_interpolation_block = FrameIntrpolationBlock(self.project_manager.models_dir)
       
        quick_memory_check("After Frame Interpolation Init")   
       
        self.postprocessing_block = PostProcessingBlock()
        
        quick_memory_check("After PostProcessing Init")
        self.first_silent_frame, self.last_silent_frame = self.preprocessing_block.get_first_last_frames(self.project_manager.silent_video_path)
        print_memory_usage("INITIALIZATION COMPLETE")
        

    def cleanup(self):

        print_memory_usage("BEFORE CLEANUP")    

        for dir in [self.project_manager.results_dir, self.project_manager.intermediate_videos_dir, self.project_manager.tmp_dir]:
            self.project_manager.clean_dir(dir)
        
        clear_gpu_cache()
        print_memory_usage("AFTER CLEANUP")
        

    def run(self, 
            ):


        print_memory_usage("RUN START")
        reset_peak_memory()  # Reset peak tracking for this run

        # Check memory before starting
        check_memory_warning()

        print("🚀 Starting lipsync execution...")
        update_peak_memory()


        self.lipsync_block.execute(
            audio_path=self.project_manager.audio_path,
            video_out_path=self.project_manager.results_dir / "output.mp4",
            tmp_audio_path=self.project_manager.tmp_dir / "modified_audio.wav", 
            guidance_scale=self.project_manager.guidance_scale

        )


        print_memory_usage("AFTER LIPSYNC EXECUTION")
        update_peak_memory()
        print("🎬 Processing frame interpolation...")


        if not self.is_first_video:
            first_frame = self.lipsync_block.get_first_gen_frame()
            intermediate_video1_path = self.project_manager.intermediate_videos_dir / "video1.mp4"
            self.frame_interpolation_block.execute(self.last_silent_frame, first_frame, self.fps, save_path=intermediate_video1_path)
            quick_memory_check("After First Frame Interpolation")
        else:
            self.is_first_video = False
            intermediate_video1_path = None


        last_frame = self.lipsync_block.get_last_gen_frame()
        intermediate_video2_path = self.project_manager.intermediate_videos_dir / "video2.mp4"
        self.frame_interpolation_block.execute(last_frame, self.first_silent_frame, self.fps, save_path=intermediate_video2_path) 

        print_memory_usage("AFTER FRAME INTERPOLATION")

        # Check memory before final processing
        if check_memory_warning():
            clear_gpu_cache()

        print("🎨 Processing postprocessing...")
        


        if intermediate_video1_path is not None:
            videos = [intermediate_video1_path, self.project_manager.results_dir / "output.mp4", intermediate_video2_path, self.project_manager.silent_video_path]
            speed_up_videos = [intermediate_video1_path, intermediate_video2_path]
        else:
            videos = [self.project_manager.results_dir / "output.mp4", intermediate_video2_path, self.project_manager.silent_video_path]
            speed_up_videos = [intermediate_video2_path]


        self.postprocessing_block.execute(videos, speed_up_videos, self.project_manager.results_dir / "final_output.mp4", 
                                          self.project_manager.background_paths, self.project_manager.save_path)
        print_memory_usage("AFTER POSTPROCESSING")
        update_peak_memory()
        
        print_peak_memory()  # Show peak usage
       
        
        self.cleanup()



if __name__ == "__main__":
        # Print system info at startup
    print_system_info()



    base_dir=Path('.')
    models_dir = Path('../checkpoints/')    
    project_manager = ProjectManager(base_dir, models_dir)


    print("🏗️  Creating App instance...")
    lipsync_app = App(project_manager)
    
    
    import time 
    print("⏱️  Starting timed execution...")
    s_time = time.time()
    audio_path = "../test_data/audios/Perfect_2_5_and_perfect_3_4_concatenated.wav"
    save_path = "../test_data/results/test1/"
    # backgrounds = ['morning', 'night', 'noon', 'sunset']
    backgrounds = ['noon']
    project_manager.set_audio_path(audio_path)
    project_manager.set_save_path(save_path)
    project_manager.set_background_paths(backgrounds)

    try:
        lipsync_app.run()    
        execution_time = time.time()
        print(execution_time-s_time)

        # Final results
        print(f"\n{'='*60}")
        print("✅ EXECUTION COMPLETE")
        print(f"{'='*60}")
        print(f"⏱️  Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
        print_memory_usage("FINAL STATE")
        
    except Exception as e:
        execution_time = time.time()
        execution_time = execution_time - s_time
        print(f"\n{'='*60}")
        print("❌ EXECUTION FAILED")
        print(f"{'='*60}")
        print(f"⏱️  Failed after: {execution_time:.2f} seconds")
        print(f"🚨 Error: {e}")
        print_memory_usage("ERROR STATE")
        raise

    finally:
        # Always show final memory state
        quick_memory_check("Script End")
















