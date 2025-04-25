import os

import librosa
import nnAudio
import numpy as np
import torch
from matplotlib import pyplot as plt
from nnAudio.features import MelSpectrogram
from pathlib import Path

from wav_to_lms import FFT_parameters, ToLogMelSpec


class RawAudioDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, csv_file, crop_frames=608, sample_rate=16000,
                 n_fft=400, hop_length=160, n_mels=80, f_min=50, f_max=8000, tfms=None, norm_stats=None):
        self.csv_dir = os.path.join(root_dir,csv_file)
        self.root_dir = root_dir
        self.file_list = self.read_csv()
        self.crop_frames = crop_frames
        self.sample_rate = sample_rate
        self.tfms = tfms
        self.norm_stats = norm_stats
        self.fmin = f_min
        self.fmax = f_max

        # Mel spectrogram extractor (matches wav_to_lms.py parameters)
        self.to_mel = nnAudio.features.MelSpectrogram(
            sr=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max,
            center=True,
            power=2,
            verbose=False,
        )

    def read_csv(self):
        # Read the CSV file and return a list of file paths
        with open(self.csv_dir, 'r') as f:
            lines = f.readlines()
        lines.pop(0)  # Remove the header line
        return lines

    def __len__(self):
        return len(self.file_list)

    def _load_and_process(self, file_path):
        wav, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)

        plt.plot(wav)
        plt.title('Signal')
        plt.xlabel('Time (samples)')
        plt.ylabel('Amplitude')

        wav_tensor = torch.from_numpy(wav.astype(np.float32))
        mel = self.to_mel(wav_tensor.unsqueeze(0))
        mel = mel.squeeze(0)
        log_mel = torch.log(mel + torch.finfo(mel.dtype).eps)

        return log_mel

    def __getitem__(self, idx):
        line = self.file_list[idx]
        file_path = os.path.join(self.root_dir, line.split(' ')[1])

        # get other info here
        log_mel = self._load_and_process(file_path)

        # Add normalization after cropping
        if self.norm_stats is not None:
            log_mel = (log_mel - self.norm_stats[0]) / self.norm_stats[1]

        # Apply transforms if defined
        if self.tfms is not None:
            log_mel = self.tfms(log_mel)

        return log_mel

    def validate_preprocessing(self, idx):
        """Compare with original wav_to_lms.py output"""
        line = self.file_list[idx]
        orig_wav_path = Path(self.root_dir) / line.split(' ')[1]

        # Get new pipeline output
        new_lms = self[idx].numpy()

        # Get original pipeline output
        prms = FFT_parameters()
        to_lms = ToLogMelSpec(prms)
        wav, _ = librosa.load(orig_wav_path, sr=prms.sample_rate, mono=True)
        orig_lms = to_lms(wav).numpy()

        # Compare
        assert np.allclose(orig_lms, new_lms, atol=1e-5), \
            f"Mismatch at {orig_wav_path}\nMax diff: {np.max(np.abs(orig_lms - new_lms))}"

        # Plot difference
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(orig_lms.squeeze(), aspect='auto', origin='lower')
        plt.title('Original Log-Mel')
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.imshow(new_lms.squeeze(), aspect='auto', origin='lower')
        plt.title('New Log-Mel')
        plt.colorbar()
        plt.tight_layout()
        plt.show()

        return np.max(np.abs(orig_lms - new_lms))

def build_dataset(cfg, mode='train'):
    """The followings configure the training dataset details.
        - data_path: Root folder of the training dataset.
        - dataset: The _name_ of the training dataset, an stem name of a `.csv` training data list.
        - norm_stats: Normalization statistics, a list of [mean, std].
        - input_size: Input size, a list of [# of freq. bins, # of time frames].
    """
    assert mode in ['train', 'val']

    transforms = None # ADD

    ds = RawAudioDataset(
        root_dir=cfg.data['root_dir'],  # Root folder containing raw `.wav` files
        csv_file=cfg.data['train_csv'] if mode=='train' else cfg.data['val_csv'],  # CSV file containing the list of files
        crop_frames=cfg.model['input_size'][1],  # Number of time frames per sample
        sample_rate=cfg.preprocessing['sample_rate'],  # Sampling rate (matches FFT_parameters)
        n_fft=cfg.preprocessing['n_fft'],  # FFT window size
        hop_length=cfg.preprocessing['hop_length'],  # Hop size
        n_mels=cfg.preprocessing['n_mels'],
        f_min=cfg.preprocessing['f_min'],
        f_max=cfg.preprocessing['f_max'],
        tfms=transforms,
        norm_stats=cfg.preprocessing['norm_stats'] if 'norm_stats' in cfg.preprocessing else None,
    )
    return ds
