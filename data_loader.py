import os
import numpy as np
import pandas as pd
from tensorflow.python.keras.utils import Sequence

from utterance_utils import mel_for_speaker_embeddings


class RandomTrainGenerator(Sequence):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __len__(self):
        return 50

    def __getitem__(self, index):
        return [np.random.randint(0, 32, (self.batch_size, 23)),
                np.random.rand(self.batch_size, 3, 160, 40),
                np.random.rand(self.batch_size, 33, 400)], \
               [np.random.rand(self.batch_size, 33, 400),
                np.random.rand(self.batch_size, 33 * 5, 1025)]


class SpeakerEmbeddingPredictionGenerator(Sequence):
    def __init__(self, numpied_dir,
                 batch_size,
                 sliding_window_size,
                 sample_rate,
                 n_fft,
                 hop_length,
                 win_length,
                 n_mels,
                 ref_db,
                 max_db
                 ):
        self.batch_size = batch_size
        self.sliding_window_size = sliding_window_size
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.ref_db = ref_db
        self.max_db = max_db

        df = pd.read_csv(os.path.join(numpied_dir, 'trans.tsv'), header=None, sep='\t')
        df['len'] = df[2].str.len()
        df = df.sort_values('len').reset_index(drop=True)
        ids = np.array(list(df[0].str.split('_')))
        self.all_utterances = os.path.abspath(numpied_dir) + '/' + pd.Series(ids[:, 0]) + '/' + \
                              pd.Series(ids[:, 1]) + '/' + df[0] + '.npy'

    def __len__(self):
        return len(self.all_utterances) // self.batch_size + 1

    def get_all_utterances(self):
        return list(self.all_utterances)

    def __getitem__(self, index):
        current_batch = self.all_utterances[index * self.batch_size: (index + 1) * self.batch_size]
        mel_specs = [
            mel_for_speaker_embeddings(utt, sample_rate=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length,
                                       win_length=self.win_length, n_mels=self.n_mels, ref_db=self.ref_db,
                                       max_db=self.max_db) for utt in current_batch]
        mel_slided = [np.stack(
            [utt[i: i + self.sliding_window_size] for i in range(0, utt.shape[0], int(self.sliding_window_size // 2)) if
             (i + self.sliding_window_size) <= utt.shape[0]]) for utt in mel_specs]
        # padding
        max_len = np.max([utt.shape[0] for utt in mel_slided])
        padded_mel_slides = np.stack(
            [np.pad(utt, ([0, max_len - utt.shape[0]], [0, 0], [0, 0]), mode='constant') for utt in mel_slided], axis=0)

        return padded_mel_slides


class SynthesizerTrainGenerator(Sequence):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
