import struct

import numpy as np
import librosa
from pydub import AudioSegment

from utterance_utils import text_to_nparray, trim_long_silences
import hparams


def inference_input(text_input, recording, max_len):
    text_int = np.concatenate([text_to_nparray(text_input), np.zeros((np.random.randint(10, 30),))])
    removed_long_silences = trim_long_silences(recording)
    try:
        utt_array, _ = librosa.effects.trim(removed_long_silences, top_db=hparams.TRIM_SILENCE_TOP_DB)
    except ValueError:
        print('No voice activity detection...')
        return removed_long_silences

    pcm_wave = struct.pack("%dh" % len(utt_array), *(np.round(utt_array * hparams.int16_max)).astype(np.int16))
    wav = AudioSegment(pcm_wave, sample_width=2, frame_rate=hparams.SAMPLE_RATE, channels=1)
    wav = wav.set_frame_rate(hparams.EMBED_SAMPLE_RATE)
    reference_utterance = librosa.util.buf_to_float(np.frombuffer(wav.raw_data, dtype=np.int16))

    if hparams.MIN_UTT_LEN >= (utt_array.shape[0] / hparams.EMBED_SAMPLE_RATE):
        repeat = (hparams.MIN_UTT_LEN * hparams.EMBED_SAMPLE_RATE) // utt_array.shape[0] + 1
        reference_utterance = np.tile(utt_array, int(repeat))

    ref_stft = librosa.stft(reference_utterance, n_fft=hparams.N_FFT, hop_length=hparams.EMBED_HOP_LENGTH, win_length=hparams.EMBED_WIN_LENGTH)
    ref_mag_spec = np.abs(ref_stft)
    ref_mel_basis = librosa.filters.mel(hparams.EMBED_SAMPLE_RATE, n_fft=hparams.N_FFT, n_mels=hparams.SPK_EMBED_N_MELS)
    ref_mel_spec = np.dot(ref_mel_basis, ref_mag_spec)

    ref_mel_db = librosa.amplitude_to_db(ref_mel_spec)
    ref_mel_db = np.clip((ref_mel_db - hparams.REF_DB + hparams.MAX_DB) / hparams.MAX_DB, 1e-8, 1)
    ref_mel_db = ref_mel_db.astype(np.float32).T

    mel_slided = np.stack(
        [ref_mel_db[i: i + hparams.SLIDING_WINDOW_SIZE] if (i + hparams.SLIDING_WINDOW_SIZE) <= ref_mel_db.shape[0] else
         ref_mel_db[-hparams.SLIDING_WINDOW_SIZE:] for i in range(0, ref_mel_db.shape[0], int(hparams.SLIDING_WINDOW_SIZE // 2))])

    dec_inputs = np.zeros((1, max_len, hparams.SYNTHESIZER_N_MELS))

    return [np.expand_dims(text_int, axis=0),
            np.expand_dims(mel_slided, axis=0),
            dec_inputs]