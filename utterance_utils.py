import copy
import librosa
import os
import numpy as np
from scipy import signal
from scipy.ndimage.morphology import binary_dilation
import struct
import webrtcvad
from pydub import AudioSegment

import hparams


def text_to_nparray(text, vocab=hparams.VOCAB):
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    return np.array([char2idx[char] for char in text if char in vocab], np.int32)


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
                    preemphasize=hparams.PREEMPHASIZE,
                    ref_db=hparams.REF_DB,
                    max_db=hparams.MAX_DB,
                    n_iter_griffin_lim=hparams.N_ITER_GRIFFIN_LIM,
                    n_fft=hparams.N_FFT,
                    hop_length=hparams.HOP_LENGTH,
                    win_length=hparams.WIN_LENGTH,
                    window=hparams.WINDOW):
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


def trim_long_silences(wav):
    """
    Ensures that segments without voice in the waveform remain no longer than a
    threshold determined by the VAD parameters in params.py.
    :param wav: the raw waveform as a numpy array of floats
    :return: the same waveform with silences trimmed away (length <= original wav length)
    """
    # Compute the voice detection window size
    samples_per_window = (hparams.vad_window_length * hparams.SAMPLE_RATE) // 1000

    # Trim the end of the audio to have a multiple of the window size
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * hparams.int16_max)).astype(np.int16))

    # Perform voice activation detection
    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=hparams.SAMPLE_RATE))
    voice_flags = np.array(voice_flags)

    # Smooth the voice detection with a moving average
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    audio_mask = moving_average(voice_flags, hparams.vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(np.bool)

    # Dilate the voiced regions
    audio_mask = binary_dilation(audio_mask, np.ones(hparams.vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)

    return wav[audio_mask == True]


def mel_for_speaker_embeddings(wav_path,
                               dataset_dir,
                               out_dir,
                               sample_rate,
                               n_fft,
                               hop_length,
                               win_length,
                               n_mels,
                               ref_db,
                               max_db):
    wav = AudioSegment.from_wav(wav_path)
    wav = wav.set_frame_rate(hparams.SAMPLE_RATE)
    utt_array = librosa.util.buf_to_float(np.frombuffer(wav.raw_data, dtype=np.int16))
    utt_array = trim_long_silences(utt_array)
    if hparams.MIN_UTT_LEN >= (utt_array.shape[0] // sample_rate) or hparams.MAX_UTT_LEN <= (
            utt_array.shape[0] // sample_rate):
        return wav_path
    npy_path = wav_path.replace(dataset_dir, out_dir)
    try:
        os.makedirs('/'.join(npy_path.split('/')[:-1]))
    except FileExistsError:
        pass
    np.save(npy_path.replace('.wav', '.npy'), utt_array)
    stft = librosa.stft(utt_array, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag_spec = np.abs(stft)
    mel_basis = librosa.filters.mel(sample_rate, n_fft=n_fft, n_mels=n_mels)
    mel_spec = np.dot(mel_basis, mag_spec)

    mel_db = librosa.amplitude_to_db(mel_spec)
    mel_db = np.clip((mel_db - ref_db + max_db) / max_db, 1e-8, 1)

    return mel_db.astype(np.float32).T, utt_array.shape[0]
