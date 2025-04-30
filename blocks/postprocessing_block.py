import subprocess
import os
import tempfile
from pathlib import Path
import ffmpeg

class PostProcessingBlock:
    def __init__(self):
        pass

    def execute(self, videos, speed_up_videos, output_file, background_paths, save_dir):
        self.transcode_all_videos(videos)
        self.speed_up_selected_videos(speed_up_videos)
        videos_paths = self.prepare_video_paths(videos, speed_up_videos)
        self.concat_videos(videos_paths, output_file)
        self.overlay_background(output_file, background_paths, save_dir)

    def run_ffmpeg_command(self, command):
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def transcode_video(self, video_path):
        output_filename = f"{os.path.splitext(video_path)[0]}_standard.mp4"
        command = [
            'ffmpeg', '-y', '-i', video_path,
            '-c:v', 'libx264', '-crf', '18', '-preset', 'slow',
            '-r', '25', '-vf', 'scale=1920:1080',
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

    def get_video_duration(self, video_path):
        try:
            probe = ffmpeg.probe(video_path)
            return float(probe["format"]["duration"])
        except Exception:
            return 0

    def speed_up_video(self, video_path):
        base_name = os.path.splitext(video_path)[0]
        input_filename = f"{base_name}_standard.mp4"
        output_filename = f"{base_name}_fast.mp4"
        silent_audio = f"{base_name}_silent.aac"
        final_output = f"{base_name}_fast_with_audio.mp4"

        self.run_ffmpeg_command([
            'ffmpeg', '-y', '-i', input_filename,
            '-filter:v', 'setpts=0.5*PTS,scale=1920:1080', '-r', '25',
            '-c:v', 'libx264', '-crf', '18', '-preset', 'slow',
            '-g', '12', '-keyint_min', '12', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', '-an', output_filename
        ])

        video_duration = self.get_video_duration(output_filename)

        self.run_ffmpeg_command([
            'ffmpeg', '-y', '-f', 'lavfi', '-i', 'anullsrc=r=48000:cl=stereo',
            '-t', str(video_duration), '-c:a', 'aac', '-b:a', '320k', silent_audio
        ])

        self.run_ffmpeg_command([
            'ffmpeg', '-y', '-i', output_filename, '-i', silent_audio,
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '320k',
            '-shortest', final_output
        ])

        return final_output

    def speed_up_selected_videos(self, videos):
        for video in videos:
            standard_video = f"{os.path.splitext(video)[0]}_standard.mp4"
            if os.path.isfile(standard_video):
                self.speed_up_video(video)

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
        return paths

    def concat_videos(self, video_paths, output_file):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', prefix='.ffmpeg_file_list_') as f:
            file_list_path = f.name
            for path in video_paths:
                f.write(f"file '{path}'\n")

        command = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', file_list_path,
            '-c:v', 'libx264', '-crf', '18', '-preset', 'slow',
            '-g', '12', '-keyint_min', '12', '-r', '25',
            '-vf', 'scale=1920:1080', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '320k', '-movflags', '+faststart', output_file
        ]
        self.run_ffmpeg_command(command)
        os.remove(file_list_path)

    def overlay_background(self, green_video, background_paths, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        tmp_dir = os.path.dirname(green_video)

        result = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", green_video
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        duration = result.stdout.strip()

        temp_transparent = os.path.join(tmp_dir, "temp_transparent.mov")
        self.run_ffmpeg_command([
            "ffmpeg", "-y", "-i", green_video,
            "-filter_complex", "chromakey=0x00FF00:0.36:0.0,format=rgba,scale=1920:1080",
            "-c:v", "qtrle", "-pix_fmt", "argb", temp_transparent
        ])

        for bg_path in background_paths:
            basename = os.path.splitext(os.path.basename(bg_path))[0]
            temp_trimmed = os.path.join(tmp_dir, f"temp_trimmed_{basename}.mp4")
            output_video = Path(save_dir) / f"{basename}.mp4"

            self.run_ffmpeg_command([
                "ffmpeg", "-y", "-i", str(bg_path), "-t", duration, "-vf", "scale=1920:1080",
                "-c:v", "libx264", "-crf", "18", "-preset", "slow", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "320k", temp_trimmed
            ])

            self.run_ffmpeg_command([
                "ffmpeg", "-y", "-i", temp_trimmed, "-i", temp_transparent,
                "-filter_complex", "[1:v]format=rgba[fg];[0:v][fg]overlay=0:0:format=auto",
                "-map", "0:a?", "-map", "0:v", "-map", "1:a?", "-shortest",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(output_video)
            ])

