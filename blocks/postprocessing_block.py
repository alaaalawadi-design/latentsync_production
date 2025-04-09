

import subprocess
import os
import ffmpeg

class PostProcessingBlock:
    def __init__(self):
        pass

    def execute(self, videos, speed_up_videos, output_file):
        self.transcode_all_videos(videos)
        self.speed_up_selected_videos(speed_up_videos)
        videos_paths = self.prepare_video_paths(videos, speed_up_videos)
        self.concat_videos(videos_paths, output_file)

    def run_ffmpeg_command(self, command):
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def transcode_video(self, video_path):
        output_filename = f"{os.path.splitext(video_path)[0]}_standard.mp4"
        command = [
            'ffmpeg', '-y', '-i', video_path,
            '-c:v', 'libx264', '-crf', '18', '-preset', 'slow', 
            '-r', '25', '-s', '1920x1088',
            '-g', '25', '-keyint_min', '25', '-sc_threshold', '0',  
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-ar', '48000', '-ac', '2', '-b:a', '320k',  
            '-movflags', '+faststart',
            output_filename
        ]
        self.run_ffmpeg_command(command)


    def transcode_all_videos(self, videos):
        for video in videos:
            if os.path.isfile(video):
                self.transcode_video(video)
            else:
                print(f"File {video} not found, skipping transcoding.")

    def get_video_duration(self, video_path):
        """Extracts the duration of a video in seconds."""
        try:
            probe = ffmpeg.probe(video_path)
            return float(probe["format"]["duration"])
        except Exception as e:
            print(f"Error getting duration for {video_path}: {e}")
            return 0

    def speed_up_video(self, video_path):
        input_filename = f"{os.path.splitext(video_path)[0]}_standard.mp4"
        output_filename = f"{os.path.splitext(video_path)[0]}_fast.mp4"
        silent_audio = f"{os.path.splitext(video_path)[0]}_silent.aac"
        final_output = f"{os.path.splitext(video_path)[0]}_fast_with_audio.mp4"

        
        command = [
            'ffmpeg', '-y', '-i', input_filename,
            '-filter:v', 'setpts=0.5*PTS',  
            '-r', '25',
            '-c:v', 'libx264', '-crf', '18', '-preset', 'slow',  
            '-g', '12', '-keyint_min', '12', '-pix_fmt', 'yuv420p',  
            '-movflags', '+faststart',  
            '-an',  
            output_filename
        ]
        self.run_ffmpeg_command(command)

        video_duration = self.get_video_duration(output_filename)
        command = [
            'ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo',
            '-t', str(video_duration), '-c:a', 'aac', '-b:a', '320k', silent_audio  
        ]
        self.run_ffmpeg_command(command)

        command = [
            'ffmpeg', '-y', '-i', output_filename, '-i', silent_audio,
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '320k',
            '-shortest', final_output
        ]
        print(f"Merging silent audio with {output_filename}...")
        self.run_ffmpeg_command(command)

        return final_output  


    def speed_up_selected_videos(self, videos):
        for video in videos:
            standard_video = f"{os.path.splitext(video)[0]}_standard.mp4"
            if os.path.isfile(standard_video):
                self.speed_up_video(video)
            else:
                print(f"Standard video {standard_video} not found, skipping speed-up.")

    def prepare_video_paths(self, videos, speed_up_videos):
        paths = []
        for video in videos:
            base_name = os.path.splitext(video)[0]
            fast_path = f"{base_name}_fast_with_audio.mp4"  
            standard_path = f"{base_name}_standard.mp4"

            if video in speed_up_videos and os.path.isfile(fast_path):
                paths.append(os.path.abspath(fast_path))
            elif os.path.isfile(standard_path):
                paths.append(os.path.abspath(standard_path))
            else:
                print(f"Neither fast nor standard file found for {video}, skipping.")
        return paths

    def concat_videos(self, video_paths, output_file):
        file_list_path = 'file.txt'
        with open(file_list_path, 'w') as f:
            for path in video_paths:
                f.write(f"file '{path}'\n")
        command = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
            '-i', file_list_path,
            '-c:v', 'libx264', '-crf', '18', '-preset', 'slow',  
            '-g', '12', '-keyint_min', '12', 
            '-r', '25', '-pix_fmt', 'yuv420p', 
            '-c:a', 'aac', '-b:a', '320k',  
            '-movflags', '+faststart',  
            output_file
        ]

        self.run_ffmpeg_command(command)



