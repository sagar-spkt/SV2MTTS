from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.optimizers import Adam

from layers import Encoder, Decoder, PostProcessing, Conditioning, InferenceSpeakerEmbedding, custom_layers
import hparams


def get_full_model(vocab_size=len(hparams.VOCAB),
                   char_embed_size=hparams.CHAR_EMBED_SIZE,
                   sliding_window_size=hparams.SLIDING_WINDOW_SIZE,
                   spk_embed_lstm_units=hparams.SPK_EMBED_LSTM_UNITS,
                   spk_embed_size=hparams.SPK_EMBED_SIZE,
                   spk_embed_num_layers=hparams.SPK_EMBED_NUM_LAYERS,
                   enc_conv1_bank_depth=hparams.ENC_CONV1_BANK_DEPTH,
                   enc_convprojec_filters1=hparams.ENC_CONVPROJEC_FILTERS1,
                   enc_convprojec_filters2=hparams.ENC_CONVPROJEC_FILTERS2,
                   enc_highway_depth=hparams.ENC_HIGHWAY_DEPTH,
                   hidden_size=hparams.HIDDEN_SIZE,
                   post_conv1_bank_depth=hparams.POST_CONV1_BANK_DEPTH,
                   post_convprojec_filters1=hparams.POST_CONVPROJEC_FILTERS1,
                   post_convprojec_filters2=hparams.POST_CONVPROJEC_FILTERS2,
                   post_highway_depth=hparams.POST_HIGHWAY_DEPTH,
                   attention_dim=hparams.ATTENTION_DIM,
                   dec_frsize=hparams.DEC_FRAME_SIZE,
                   target_size=hparams.TARGET_MAG_FRAME_SIZE,
                   n_mels=hparams.SYNTHESIZER_N_MELS,
                   embed_mels=hparams.SPK_EMBED_N_MELS,
                   enc_seq_len=None,
                   dec_seq_len=None
                   ):
    char_inputs = Input(shape=(enc_seq_len,), name='char_inputs')
    decoder_inputs = Input(shape=(dec_seq_len, dec_frsize), name='decoder_inputs')
    spk_inputs = Input(shape=(None, sliding_window_size, embed_mels), name='spk_embed_inputs')

    char_encoder = Encoder(hidden_size=hidden_size // 2,
                           vocab_size=vocab_size,
                           embedding_size=char_embed_size,
                           conv1d_bank_depth=enc_conv1_bank_depth,
                           convprojec_filters1=enc_convprojec_filters1,
                           convprojec_filters2=enc_convprojec_filters2,
                           highway_depth=enc_highway_depth,
                           name='char_encoder')
    speaker_encoder = InferenceSpeakerEmbedding(lstm_units=spk_embed_lstm_units,
                                                proj_size=spk_embed_size,
                                                num_layers=spk_embed_num_layers,
                                                trainable=False,
                                                name='embeddings')
    condition = Conditioning()
    decoder = Decoder(hidden_size=hidden_size,
                      attention_dim=attention_dim,
                      dec_output_size=dec_frsize,
                      name='decoder')
    post_processing = PostProcessing(hidden_size=hidden_size // 2,
                                     conv1d_bank_depth=post_conv1_bank_depth,
                                     convprojec_filters1=post_convprojec_filters1,
                                     convprojec_filters2=post_convprojec_filters2,
                                     highway_depth=post_highway_depth,
                                     dec_frsize=dec_frsize,
                                     target_frsize=target_size,
                                     dec_frreshape=n_mels,
                                     name='postprocessing')

    char_enc = char_encoder(char_inputs)
    spk_embed = speaker_encoder(spk_inputs)
    conditioned_char_enc = condition([char_enc, spk_embed])
    decoder_pred, alignments = decoder([conditioned_char_enc, decoder_inputs], initial_state=None)
    postnet_out = post_processing(decoder_pred)

    full_model = Model(inputs=[char_inputs, spk_inputs, decoder_inputs],
                       outputs=[decoder_pred, postnet_out, alignments])
    return full_model


def get_speaker_embedding_model(sliding_window_size=hparams.SLIDING_WINDOW_SIZE,
                                embed_mels=hparams.SPK_EMBED_N_MELS,
                                spk_embed_lstm_units=hparams.SPK_EMBED_LSTM_UNITS,
                                spk_embed_size=hparams.SPK_EMBED_SIZE,
                                spk_embed_num_layers=hparams.SPK_EMBED_NUM_LAYERS):
    spk_inputs = Input(shape=(None, sliding_window_size, embed_mels), name='spk_embed_inputs')
    speaker_encoder = InferenceSpeakerEmbedding(lstm_units=spk_embed_lstm_units,
                                                proj_size=spk_embed_size,
                                                num_layers=spk_embed_num_layers,
                                                trainable=False,
                                                name='embeddings')
    spk_embed = speaker_encoder(spk_inputs)
    speaker_embedding_model = Model(inputs=[spk_inputs],
                                    outputs=[spk_embed])
    return speaker_embedding_model


def get_synthesizer_model(vocab_size=len(hparams.VOCAB),
                          char_embed_size=hparams.CHAR_EMBED_SIZE,
                          spk_embed_size=hparams.SPK_EMBED_SIZE,
                          enc_conv1_bank_depth=hparams.ENC_CONV1_BANK_DEPTH,
                          enc_convprojec_filters1=hparams.ENC_CONVPROJEC_FILTERS1,
                          enc_convprojec_filters2=hparams.ENC_CONVPROJEC_FILTERS2,
                          enc_highway_depth=hparams.ENC_HIGHWAY_DEPTH,
                          hidden_size=hparams.HIDDEN_SIZE,
                          post_conv1_bank_depth=hparams.POST_CONV1_BANK_DEPTH,
                          post_convprojec_filters1=hparams.POST_CONVPROJEC_FILTERS1,
                          post_convprojec_filters2=hparams.POST_CONVPROJEC_FILTERS2,
                          post_highway_depth=hparams.POST_HIGHWAY_DEPTH,
                          dec_frsize=hparams.DEC_FRAME_SIZE,
                          attention_dim=hparams.ATTENTION_DIM,
                          target_size=hparams.TARGET_MAG_FRAME_SIZE,
                          n_mels=hparams.SYNTHESIZER_N_MELS,
                          learning_rate=hparams.LEARNING_RATE,
                          clipnorm=hparams.CLIPNORM,
                          enc_seq_len=None,
                          dec_seq_len=None):
    char_inputs = Input(shape=(enc_seq_len,), name='char_inputs')
    decoder_inputs = Input(shape=(dec_seq_len, dec_frsize), name='decoder_inputs')
    spk_embed_inputs = Input(shape=(spk_embed_size,), name='spk_embed_inputs')

    char_encoder = Encoder(hidden_size=hidden_size // 2,
                           vocab_size=vocab_size,
                           embedding_size=char_embed_size,
                           conv1d_bank_depth=enc_conv1_bank_depth,
                           convprojec_filters1=enc_convprojec_filters1,
                           convprojec_filters2=enc_convprojec_filters2,
                           highway_depth=enc_highway_depth,
                           name='char_encoder')
    condition = Conditioning()
    decoder = Decoder(hidden_size=hidden_size,
                      attention_dim=attention_dim,
                      dec_output_size=dec_frsize,
                      name='decoder')
    post_processing = PostProcessing(hidden_size=hidden_size // 2,
                                     conv1d_bank_depth=post_conv1_bank_depth,
                                     convprojec_filters1=post_convprojec_filters1,
                                     convprojec_filters2=post_convprojec_filters2,
                                     highway_depth=post_highway_depth,
                                     dec_frsize=dec_frsize,
                                     target_frsize=target_size,
                                     dec_frreshape=n_mels,
                                     name='postprocessing')

    char_enc = char_encoder(char_inputs)
    conditioned_char_enc = condition([char_enc, spk_embed_inputs])
    decoder_pred, alignments = decoder([conditioned_char_enc, decoder_inputs], initial_state=None)
    postnet_out = post_processing(decoder_pred)

    synthesizer_model = Model(inputs=[char_inputs, spk_embed_inputs, decoder_inputs],
                              outputs=[decoder_pred, postnet_out, alignments])
    optimizer = Adam(lr=learning_rate, clipnorm=clipnorm)
    synthesizer_model.compile(optimizer=optimizer, loss=['mae', 'mae', None], loss_weights=[1., 1., None])

    return synthesizer_model


def load_saved_model(model_path):
    return load_model(model_path, custom_objects=custom_layers)
