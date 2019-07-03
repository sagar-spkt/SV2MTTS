import os
import glob
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_loader import SpeakerEmbeddingPredictionGenerator
from models import get_speaker_embedding_model


def wav_to_numpy(dataset_rootdir, preprocess_rootdir, sample_rate, min_len, max_len):
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

    for utterance in tqdm(all_utterances):
        y, _ = librosa.load(utterance, sr=sample_rate)
        y, _ = librosa.effects.trim(y)
        if len(y) < int(min_len * sample_rate) or len(y) > int(max_len * sample_rate):
            continue
        total_time += len(y) / sample_rate
        len_satisfied_utterances.append(utterance.split('/')[-1].replace('.wav', ''))  # only the name in dataframe
        savepath = os.path.join(preprocess_rootdir, '/'.join(utterance.split('/')[-3:]).replace('wav', 'npy'))
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        np.save(savepath, y)

    transcripts = transcripts[transcripts[0].isin(len_satisfied_utterances)]
    transcripts.to_csv(os.path.join(preprocess_rootdir, 'trans.tsv'), sep='\t', header=False, index=False)
    with open(os.path.join(preprocess_rootdir, 'logs.txt'), 'w') as f:
        f.write(str(total_time))
    print('Done')


def wav_to_speaker_embeddings(numpied_dir,
                              model_path,
                              batch_size,
                              sample_rate,
                              n_fft,
                              hop_length,
                              win_length,
                              n_mels,
                              ref_db,
                              max_db,
                              sliding_window_size,
                              spk_embed_lstm_units,
                              spk_embed_size,
                              spk_embed_num_layers,
                              verbose=1):
    speaker_embedding_model = get_speaker_embedding_model(sliding_window_size=sliding_window_size,
                                                          embed_mels=n_mels,
                                                          spk_embed_lstm_units=spk_embed_lstm_units,
                                                          spk_embed_size=spk_embed_size,
                                                          spk_embed_num_layers=spk_embed_num_layers)
    speaker_generator = SpeakerEmbeddingPredictionGenerator(numpied_dir,
                                                            batch_size=batch_size,
                                                            sliding_window_size=sliding_window_size,
                                                            sample_rate=sample_rate,
                                                            n_fft=n_fft,
                                                            hop_length=hop_length,
                                                            win_length=win_length,
                                                            n_mels=n_mels,
                                                            ref_db=ref_db,
                                                            max_db=max_db)
    speaker_embedding_model.load_weights(model_path, by_name=True)
    speaker_embeddings = speaker_embedding_model.predict_generator(speaker_generator, verbose=verbose)

    iterator = tqdm(zip(speaker_generator.get_all_utterances(), speaker_embeddings)) if verbose else zip(
        speaker_generator.get_all_utterances(), speaker_embeddings)

    for utterance, embedding in iterator:
        np.save(utterance.replace('.npy', '_embed.npy'), embedding)
