import copy
import librosa
import numpy as np
from scipy import signal

import hparams


def text_to_nparray(text, vocab=hparams.VOCAB):
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    return np.array([char2idx[char] for char in text], np.int32)


def get_spectrograms(waveform,
                     sample_rate,
                     preemphasize,
                     hop_length,
                     win_length,
                     n_fft,
                     window,
                     n_mels,
                     ref_db,
                     max_db):
    waveform = signal.lfilter([1, -preemphasize], [1], waveform)
    spec = librosa.stft(y=waveform,
                        n_fft=n_fft,
                        hop_length=hop_length,
                        win_length=win_length,
                        window=window)
    mag_spec = np.abs(spec)

    mel_basis = librosa.filters.mel(sr=sample_rate,
                                    n_fft=n_fft,
                                    n_mels=n_mels)
    mel_spec = np.dot(mel_basis, mag_spec)

    mag_db = librosa.amplitude_to_db(mag_spec)
    mel_db = librosa.amplitude_to_db(mel_spec)

    mag_spec = np.clip((mag_db - ref_db + max_db) / max_db, 1e-8, 1)
    mel_spec = np.clip((mel_db - ref_db + max_db) / max_db, 1e-8, 1)

    mag_spec = mag_spec.astype(np.float32).T
    mel_spec = mel_spec.astype(np.float32).T

    return mag_spec, mel_spec


def griffin_lim(mag_spectro,
                n_iter_griffin_lim,
                n_fft,
                hop_length,
                win_length,
                window):
    spectro = copy.deepcopy(mag_spectro)
    for i in range(n_iter_griffin_lim):
        estimated_wav = librosa.istft(spectro,
                                      hop_length=hop_length,
                                      win_length=win_length,
                                      window=window)
        estimated_stft = librosa.stft(estimated_wav,
                                      n_fft=n_fft,
                                      hop_length=hop_length,
                                      win_length=win_length,
                                      window=window)
        phase = estimated_stft / np.maximum(1e-8, np.abs(estimated_stft))
        spectro = mag_spectro * phase
    estimated_wav = librosa.istft(spectro,
                                  hop_length=hop_length,
                                  win_length=win_length,
                                  window=window)
    return np.real(estimated_wav)


def mag_spectro2wav(mag_spectro,
                    preemphasize,
                    ref_db,
                    max_db,
                    n_iter_griffin_lim,
                    n_fft=None,
                    hop_length=None,
                    win_length=None,
                    window=None):
    mag_spectro = mag_spectro.T
    mag_spectro = (np.clip(mag_spectro, 0, 1) * max_db) - max_db + ref_db
    mag_spectro = np.power(10.0, mag_spectro / 20)
    wav = griffin_lim(mag_spectro,
                      n_iter_griffin_lim=n_iter_griffin_lim,
                      n_fft=n_fft,
                      hop_length=hop_length,
                      win_length=win_length,
                      window=window)
    wav = signal.lfilter([1], [1, -preemphasize], wav)
    wav, _ = librosa.effects.trim(wav)
    return wav.astype(np.float32)


def mel_for_speaker_embeddings(numpy_path,
                               sample_rate,
                               n_fft,
                               hop_length,
                               win_length,
                               n_mels,
                               ref_db,
                               max_db):
    y = np.load(numpy_path)

    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag_spec = np.abs(stft)
    mel_basis = librosa.filters.mel(sample_rate, n_fft=n_fft, n_mels=n_mels)
    mel_spec = np.dot(mel_basis, mag_spec)

    mel_db = librosa.amplitude_to_db(mel_spec)
    mel_db = np.clip((mel_db - ref_db + max_db) / max_db, 1e-8, 1)

    return mel_db.astype(np.float32).T
