import numpy as np 
from tflibrosa import STFT, Spectrogram, LogmelFilterBank
import librosa
import tensorflow as tf 
audio = np.random.uniform(0,1 ,(32000 * 5))
print(audio.shape)

sample_rate = 32000
n_fft = 2048
hop_size = 512
window = 'hann'
pad_mode = 'reflect'
mel_bins = 64
ref = 1.0
amin = 1e-10
fmin = 20
fmax = 16000 
top_db = 80.0
center = True 
dtype=None

def mel_spectrogram_librosa(data, n_fft, hop_size, window, center, dtype, pad_mode, fmin, fmax, sample_rate, n_mels, top_db, ref, amin):

    np_stft_matrix = librosa.stft(y=data, n_fft=n_fft, hop_length=hop_size,
            win_length=n_fft, window=window, center=center, dtype=dtype,
            pad_mode=pad_mode)

 
    np_melW = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels,
        fmin=fmin, fmax=fmax).T

    np_mel_spectrogram = np.dot(np.abs(np_stft_matrix.T) ** 2, np_melW)

    np_logmel_spectrogram = librosa.power_to_db(
        np_mel_spectrogram, ref=ref, amin=amin, top_db=top_db)

    return np_logmel_spectrogram


if __name__ == "__main__":
    spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_size, 
                win_length=n_fft, window=window, center=center, pad_mode=pad_mode, 
                freeze_parameters=True, dtype="float32")

    # Logmel feature extractor
    logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft, is_log=True, 
        n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
        freeze_parameters=True, dtype="float32")


    spectrogram = spectrogram_extractor(audio[None, :])
    mel_spectrogram = tf.squeeze(logmel_extractor(spectrogram), axis=0)

    mel_spectrogram2 = mel_spectrogram_librosa( audio, n_fft, hop_size, window, center, dtype, pad_mode, fmin, fmax, sample_rate, mel_bins, top_db, ref, amin)
    print(mel_spectrogram.shape, mel_spectrogram2.shape)
    print(mel_spectrogram2[0,:15])
    print(mel_spectrogram[0,:15])
