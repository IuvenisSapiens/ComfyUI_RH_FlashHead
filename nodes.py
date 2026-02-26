import comfy.utils

import folder_paths
try:
    from comfy_api.input_impl.video_types import VideoFromFile
except ImportError:
    VideoFromFile = None
from pathlib import Path
from optimum.quanto import freeze, qint8, quantize 
import uuid

import os
import numpy as np
import time
import torch
import torch.distributed as dist
import subprocess
import imageio
import librosa
import numpy as np
from loguru import logger
from collections import deque
from datetime import datetime
from PIL import Image
import io
import av
import yaml

# from .flash_head.inference import get_pipeline, get_base_data, get_infer_params, get_audio_embedding, run_pipeline
from .flash_head.src.pipeline.flash_head_pipeline import FlashHeadPipeline
from .flash_head.inference import get_audio_embedding, run_pipeline

class RunningHub_FlashHead_Loader:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": (["pro", "lite"], {"default": "lite"}),
            }
        }

    RETURN_TYPES = ('RH_FlashHead_Pipeline', )
    RETURN_NAMES = ('FlashHead Pipeline', )
    FUNCTION = "load"
    CATEGORY = "RunningHub/FlashHead"

    def load(self, model_type):

        ckpt_dir = os.path.join(folder_paths.models_dir, 'Soul-AILab', 'SoulX-FlashHead-1_3B')
        wav2vec_dir = os.path.join(folder_paths.models_dir, 'wav2vec', 'facebook', 'wav2vec2-base-960h')

        pipeline = FlashHeadPipeline(
            checkpoint_dir=ckpt_dir,
            model_type=model_type,
            wav2vec_dir=wav2vec_dir,
            device='cuda',
            use_usp=False,
        )
        return (pipeline, )

class RunningHub_FlashHead_Sampler:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("RH_FlashHead_Pipeline", ),
                "ref_audio": ("AUDIO", ),
                "avatar_image": ("IMAGE", ),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "width": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 8}),
            },
        }

    RETURN_TYPES = ('VIDEO',)
    RETURN_NAMES = ('video',)
    FUNCTION = "sample"
    CATEGORY = "RunningHub/FlashHead"

    def save_wav(self, ref_audio, save_path):
        waveform = ref_audio['waveform'][0]
        layout = 'mono' if waveform.shape[0] == 1 else 'stereo'
        output_buffer = io.BytesIO()
        output_container = av.open(output_buffer, mode='w', format='flac')
        out_stream = output_container.add_stream("flac", rate=ref_audio['sample_rate'], layout=layout)
        frame = av.AudioFrame.from_ndarray(waveform.movedim(0, 1).reshape(1, -1).float().numpy(), format='flt', layout=layout)
        frame.sample_rate = ref_audio['sample_rate']
        frame.pts = 0
        output_container.mux(out_stream.encode(frame))

        # Flush encoder
        output_container.mux(out_stream.encode(None))

        # Close containers
        output_container.close()

        # Write the output to file
        output_buffer.seek(0)
        with open(save_path, 'wb') as f:
            f.write(output_buffer.getbuffer())

    def save_avatar_image(self, avatar_image, save_path):
        i = 255. * avatar_image.squeeze().cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        img.save(save_path)

    def save_video(self, frames_list, video_path, audio_path, fps):
        temp_video_path = video_path.replace('flashtalk_', 'flashtalk_temp_')
        with imageio.get_writer(temp_video_path, format='mp4', mode='I',
                                fps=fps , codec='h264', ffmpeg_params=['-bf', '0']) as writer:
            for frames in frames_list:
                frames = frames.numpy().astype(np.uint8)
                for i in range(frames.shape[0]):
                    frame = frames[i, :, :, :]
                    writer.append_data(frame)
        
        # merge video and audio
        # cmd = ['ffmpeg', '-i', temp_video_path, '-i', audio_path, '-c', 'copy', '-shortest', video_path, '-y']
        cmd = ['ffmpeg', '-i', temp_video_path, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', '-shortest', video_path, '-y']
        subprocess.run(cmd)
        os.remove(temp_video_path)

    def sample(self, **kwargs):
        ref_audio = kwargs.get('ref_audio')
        tmp_wav_path = os.path.join(folder_paths.get_temp_directory(), f"flashtalk_audio_{uuid.uuid4()}.wav")
        self.save_wav(ref_audio, tmp_wav_path)

        avatar_image = kwargs.get('avatar_image')
        tmp_avatar_image_path = os.path.join(folder_paths.get_temp_directory(), f"flashtalk_avatar_image_{uuid.uuid4()}.png")
        self.save_avatar_image(avatar_image, tmp_avatar_image_path)

        infer_params_path = Path(__file__).resolve().parent / "flash_head" / "configs" / "infer_params.yaml"
        with open(infer_params_path, "r") as f:
            infer_params = yaml.safe_load(f)

        pipeline = kwargs.get('pipeline')
        motion_frames_latent_num = infer_params['motion_frames_latent_num']
        motion_frames_num = (motion_frames_latent_num - 1) * pipeline.config.vae_stride[0] + 1
        infer_params['motion_frames_num'] = motion_frames_num

        base_seed = kwargs.get('seed') ^ (2 ** 32)
        width = kwargs.get('width', infer_params['width'])
        height = kwargs.get('height', infer_params['height'])
        infer_params['width'] = width
        infer_params['height'] = height

        #kiki
        use_face_crop = False

        # TODO: move to args
        if pipeline.model_type == "pretrained":
            infer_params['sample_steps'] = 20
        else:
            infer_params['sample_steps'] = 4

        pipeline.prepare_params(
            cond_image_path_or_dir=tmp_avatar_image_path,
            target_size=(infer_params['height'], infer_params['width']),
            frame_num=infer_params['frame_num'],
            motion_frames_num=infer_params['motion_frames_num'],
            sampling_steps=infer_params['sample_steps'],
            seed=base_seed,
            shift=infer_params['sample_shift'],
            color_correction_strength=infer_params['color_correction_strength'],
            use_face_crop=use_face_crop,
        )

        sample_rate = infer_params['sample_rate']
        tgt_fps = infer_params['tgt_fps']
        cached_audio_duration = infer_params['cached_audio_duration']
        frame_num = infer_params['frame_num']
        motion_frames_num = infer_params['motion_frames_num']
        slice_len = frame_num - motion_frames_num

        human_speech_array_all, _ = librosa.load(tmp_wav_path, sr=infer_params['sample_rate'], mono=True)

        cached_audio_length_sum = sample_rate * cached_audio_duration
        audio_end_idx = cached_audio_duration * tgt_fps
        audio_start_idx = audio_end_idx - frame_num

        audio_dq = deque([0.0] * cached_audio_length_sum, maxlen=cached_audio_length_sum)

        human_speech_array_slice_len = slice_len * sample_rate // tgt_fps
        human_speech_array_slices = human_speech_array_all[:(len(human_speech_array_all)//(human_speech_array_slice_len))*human_speech_array_slice_len].reshape(-1, human_speech_array_slice_len)

        generated_list = []
        chunk_num = len(human_speech_array_slices)
        self.pbar = comfy.utils.ProgressBar(chunk_num)
        for chunk_idx, human_speech_array in enumerate(human_speech_array_slices):
            torch.cuda.synchronize()
            start_time = time.time()

            # streaming encode audio chunks
            audio_dq.extend(human_speech_array.tolist())
            audio_array = np.array(audio_dq)
            audio_embedding = get_audio_embedding(pipeline, audio_array, audio_start_idx, audio_end_idx, infer_params=infer_params)

            # inference
            video = run_pipeline(pipeline, audio_embedding)

            torch.cuda.synchronize()
            end_time = time.time()
            logger.info(f"Generate video chunk-{chunk_idx} done, cost time: {(end_time - start_time):.3f}s")

            generated_list.append(video.cpu())
            self.pbar.update(1)
        output_path = os.path.join(folder_paths.get_output_directory(), f"flashtalk_video_{uuid.uuid4()}.mp4")
        self.save_video(generated_list, output_path, tmp_wav_path, fps=tgt_fps)
        return (self.create_video_object(output_path), )

    def create_video_object(self, video_path):
        """Create ComfyUI VIDEO object"""
        if VideoFromFile is not None:
            return VideoFromFile(video_path)
        else:
            # Fallback: return file path as string
            return video_path

    def update(self):
        self.pbar.update(1)

NODE_CLASS_MAPPINGS = {
    "RunningHub SoulX-FlashHead Loader": RunningHub_FlashHead_Loader,
    "RunningHub SoulX-FlashHead Sampler": RunningHub_FlashHead_Sampler,
}