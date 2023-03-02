import os
import sys

import librosa
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AudioDataset
from model import UNet

batch_size = 1


def main():
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet()
    model.load_state_dict(
        torch.load("itmo_denoise/models/best_model.pt", map_location=device)
    )
    model.eval()
    model.to(device)

    with torch.no_grad():
        path = os.path.join(sys.path[0], "./user_input/")
        infer_dataset = AudioDataset(path, None)
        infer_dataloader = DataLoader(
            infer_dataset, batch_size=batch_size, shuffle=True
        )

        for batch in tqdm(infer_dataloader):
            _, clean_mel = batch
            clean_mel = clean_mel.to(device)
            clean_mel = clean_mel.unsqueeze(1)  # Add channel dimension
            reconstructed_mel = model(clean_mel).squeeze(1).numpy()

            melspectrogram = librosa.db_to_power(reconstructed_mel, ref=1.0)

            res = librosa.feature.inverse.mel_to_audio(
                melspectrogram,
                sr=16_000,
                n_fft=2048,
                hop_length=512,
                win_length=None,
                window="hann",
                center=True,
                pad_mode="reflect",
                power=2.0,
                n_iter=32,
            )

            sf.write("itmo_denouse/user_input/stereo_file.wav", res, 16000)


if __name__ == "__main__":
    main()
