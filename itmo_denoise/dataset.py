import os
import random

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


class AudioTransform:
    def __init__(self, probability=0.5, sr=16000):
        self.probability = probability
        self.sr = sr

    def __call__(self, sample):
        if random.random() < self.probability:
            sample = self.time_shift(sample)
        if random.random() < self.probability:
            sample = self.pitch_shift(sample)
        return sample

    def time_shift(self, sample):
        shift = np.random.randint(self.sr // 10)
        if shift == 0:
            return sample
        elif shift > 0:
            sample = np.pad(sample, (shift, 0), mode="constant")[:-shift]
        else:
            sample = np.pad(sample, (0, -shift), mode="constant")[-shift:]
        return sample

    def pitch_shift(self, sample):
        n_steps = np.random.randint(-4, 4)
        if n_steps == 0:
            return sample
        sample = librosa.effects.pitch_shift(sample, sr=self.sr, n_steps=n_steps)
        return sample


class AudioDataset(Dataset):
    def __init__(self, data_dir, transform=None, crop_size=284):
        self.data_dir = data_dir
        self.transform = transform
        self.crop_size = crop_size

        self.audio_files = [
            os.path.join(self.data_dir, file)
            for file in os.listdir(self.data_dir)
            if file.endswith(".wav")
        ]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        audio, sr = librosa.load(audio_path, sr=16000)

        # Apply augmentations
        if self.transform:
            audio = self.transform(audio)

        # Convert audio to melspectrogram
        melspectrogram = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=128, fmax=8000
        )
        melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)

        # Crop or pad melspectrogram to fixed size
        if melspectrogram.shape[1] > self.crop_size:
            start = np.random.randint(0, melspectrogram.shape[1] - self.crop_size)
            melspectrogram = melspectrogram[:, start : start + self.crop_size]
        else:
            pad_width = self.crop_size - melspectrogram.shape[1]
            melspectrogram = np.pad(
                melspectrogram, ((0, 0), (0, pad_width)), mode="constant"
            )

        # Get noisy melspectrogram
        noisy_melspectrogram = melspectrogram.copy()

        # Apply noise augmentation
        if self.transform:
            noisy_audio = self.transform(audio)
            noisy_melspectrogram = librosa.feature.melspectrogram(
                y=noisy_audio, sr=sr, n_mels=128, fmax=8000
            )
            noisy_melspectrogram = librosa.power_to_db(noisy_melspectrogram, ref=np.max)

            # Crop or pad noisy melspectrogram to fixed size
            if noisy_melspectrogram.shape[1] > self.crop_size:
                start = np.random.randint(
                    0, noisy_melspectrogram.shape[1] - self.crop_size
                )
                noisy_melspectrogram = noisy_melspectrogram[
                    :, start : start + self.crop_size
                ]
            else:
                pad_width = self.crop_size - noisy_melspectrogram.shape[1]
                noisy_melspectrogram = np.pad(
                    noisy_melspectrogram, ((0, 0), (0, pad_width)), mode="constant"
                )

        return noisy_melspectrogram, melspectrogram
