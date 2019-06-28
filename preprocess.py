import os
import glob
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm


def wav_to_numpy(dataset_rootdir, preprocess_rootdir, sample_rate, min_len):
    dataset_rootdir = os.path.abspath(dataset_rootdir)
    preprocess_rootdir = os.path.abspath(preprocess_rootdir)

    trans_rows = []
    for trans_file in glob.glob(os.path.join(dataset_rootdir, '*/*/*.trans.tsv')):
        with open(trans_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                trans_rows.append(line[:-1].split('\t'))
    transcripts = pd.DataFrame(trans_rows)

    all_utterances = glob.glob(os.path.join(dataset_rootdir, '*/*/*.wav'))
    len_satisfied_utterances = []
    total_time = 0
    max_len = 0

    for utterance in tqdm(all_utterances):
        y, _ = librosa.load(utterance, sr=sample_rate)
        y, _ = librosa.effects.trim(y)
        if len(y) < int(min_len*sample_rate):
            continue
        total_time += len(y) / sample_rate
        max_len = np.max((max_len, len(y)))
        len_satisfied_utterances.append(utterance.split('/')[-1].replace('.wav', ''))  # only the name in dataframe
        savepath = os.path.join(preprocess_rootdir, '/'.join(utterance.split('/')[-3:]).replace('wav', 'npy'))
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        np.save(savepath, y)

    transcripts = transcripts[transcripts[0].isin(len_satisfied_utterances)]
    transcripts.to_csv(os.path.join(preprocess_rootdir, 'trans.tsv'), sep='\t', header=False, index=False)
    with open(os.path.join(preprocess_rootdir, 'logs.txt'), 'w') as f:
        f.write(str(total_time) + '\t' + str(max_len))
    print('Done')

