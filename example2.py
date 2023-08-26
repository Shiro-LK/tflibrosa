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

 

if __name__ == "__main__":
    spectrogram_extractor = Spectrogram(n_fft=n_fft, hop_length=hop_size, 
                win_length=n_fft, window=window, center=center, pad_mode=pad_mode, 
                freeze_parameters=True, dtype="float32")

    # Logmel feature extractor
    logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft, is_log=True, 
        n_mels=mel_bins, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, 
        freeze_parameters=True, dtype="float32")


    inputs = tf.keras.layers.Input((None, ))
    x = spectrogram_extractor(inputs)
    x = logmel_extractor(x)
    x = tf.concat((x, x, x), axis=-1)
    print(x.shape)
    backbone = tf.keras.applications.vgg16.VGG16(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=None,
            pooling="avg",
            classes=1000,
            classifier_activation='softmax'
        )

    output = backbone(x)
    print(output.shape)
    model = tf.keras.Model(inputs=inputs, outputs=output)

    out = model.predict(audio.reshape(1,-1))
    print(out.shape)
    print(model.summary())