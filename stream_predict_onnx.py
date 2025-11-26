#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import os
from typing import Tuple

import numpy as np
import soundfile as sf

from funasr_onnx.paraformer_online_bin import Paraformer


def _load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio from disk and resample/convert if needed."""
    speech, sample_rate = sf.read(path, dtype="float32")
    if speech.ndim > 1:
        speech = speech.mean(axis=1)
    if sample_rate != target_sr:
        import librosa

        speech = librosa.resample(speech, orig_sr=sample_rate, target_sr=target_sr)
        sample_rate = target_sr
    return speech.astype(np.float32), sample_rate


def main():
    chunk_size = [0, 10, 5]  # [0, 10, 5] -> 600 ms window; second entry controls stride

    model_dir = os.path.join("models", "speech_paraformer-large_asr_onnx")
    wav_file = "创建警单.wav"

    paraformer = Paraformer(model_dir=model_dir, chunk_size=chunk_size, device_id="-1")

    speech, sample_rate = _load_audio(wav_file)
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz audio, but got {sample_rate} Hz after resampling.")

    chunk_stride = chunk_size[1] * 960  # 960 samples ~= 60 ms (10 frames) @16 kHz
    total_chunk_num = int((len(speech) - 1) / chunk_stride + 1)

    cache = {}
    param_dict = {"cache": cache, "is_final": False}

    for idx in range(total_chunk_num):
        start = idx * chunk_stride
        end = (idx + 1) * chunk_stride
        speech_chunk = speech[start:end]
        if not len(speech_chunk):
            continue
        param_dict["is_final"] = idx == total_chunk_num - 1
        results = paraformer(speech_chunk, param_dict=param_dict)
        for result in results or []:
            print(result.get("preds", ""))


if __name__ == "__main__":
    main()

