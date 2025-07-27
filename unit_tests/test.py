
import sys
from pathlib import Path

# ----- IMPORT AI ----------
sys.path.append('/media/administrator/disk_2/GenAI2/latentsync_production/cysync')
import main

# --------- MODEL LOADING ----------
from projectmanager import ProjectManager

base_dir = Path('/media/administrator/disk_2/GenAI2/latentsync_production/cysync')
models_dir = Path('/media/administrator/disk_2/GenAI2/LatentSync/checkpoints')
project_manager = ProjectManager(base_dir, models_dir)

lipsync_app = main.App(project_manager)



# --------- MODEL INFERENCE ----------
audio_path = "/media/administrator/disk_2/GenAI2/latentsync_production/cysync/test_data/audios/audio1.wav"
save_path = "/media/administrator/disk_2/GenAI2/latentsync_production/cysync/test_data/results/test1/"
backgrounds = ['morning', 'night', 'noon', 'sunset']
project_manager.set_audio_path(audio_path)
project_manager.set_save_path(save_path)
project_manager.set_background_paths(backgrounds)
lipsync_app.run()    


audio_path = "/media/administrator/disk_2/GenAI2/latentsync_production/cysync/test_data/audios/audio2.wav"
save_path = "/media/administrator/disk_2/GenAI2/latentsync_production/cysync/test_data/results/test2/"
backgrounds = ['morning', 'night', 'noon', 'sunset']
project_manager.set_audio_path(audio_path)
project_manager.set_save_path(save_path)
project_manager.set_background_paths(backgrounds)
lipsync_app.run()    



audio_path = "/media/administrator/disk_2/GenAI2/latentsync_production/cysync/test_data/audios/perfect1_2.wav"
save_path = "/media/administrator/disk_2/GenAI2/latentsync_production/cysync/test_data/results/test3/"
backgrounds = ['morning', 'night', 'noon', 'sunset']
project_manager.set_audio_path(audio_path)
project_manager.set_save_path(save_path)
project_manager.set_background_paths(backgrounds)
lipsync_app.run()    



audio_path = "/media/administrator/disk_2/GenAI2/latentsync_production/cysync/test_data/audios/Perfect_2_5.wav"
save_path = "/media/administrator/disk_2/GenAI2/latentsync_production/cysync/test_data/results/test4/"
backgrounds = ['morning', 'night', 'noon', 'sunset']
project_manager.set_audio_path(audio_path)
project_manager.set_save_path(save_path)
project_manager.set_background_paths(backgrounds)
lipsync_app.run()    



audio_path = "/media/administrator/disk_2/GenAI2/latentsync_production/cysync/test_data/audios/perfect_3_4.wav"
save_path = "/media/administrator/disk_2/GenAI2/latentsync_production/cysync/test_data/results/test5/"
backgrounds = ['morning', 'night', 'noon', 'sunset']
project_manager.set_audio_path(audio_path)
project_manager.set_save_path(save_path)
project_manager.set_background_paths(backgrounds)
lipsync_app.run()    


audio_path = "/media/administrator/disk_2/GenAI2/latentsync_production/cysync/test_data/audios/perfect4.wav"
save_path = "/media/administrator/disk_2/GenAI2/latentsync_production/cysync/test_data/results/test6/"
backgrounds = ['morning', 'night', 'noon', 'sunset']
project_manager.set_audio_path(audio_path)
project_manager.set_save_path(save_path)
project_manager.set_background_paths(backgrounds)
lipsync_app.run()    
