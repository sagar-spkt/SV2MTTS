from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.optimizers import Adam

from layers import Encoder, Decoder, PostProcessing, Conditioning, InferenceSpeakerEmbedding, custom_layers, \
    TestSpeakerEmbedding, TestSpeakerSimilarity
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
                   target_size=hparams.TARGET_MAG_FRAME_SIZE,
                   n_mels=hparams.SYNTHESIZER_N_MELS,
                   output_per_step=hparams.OUTPUT_PER_STEP,
                   embed_mels=hparams.SPK_EMBED_N_MELS,
                   enc_seq_len=None,
                   dec_seq_len=None
                   ):
    char_inputs = Input(shape=(enc_seq_len,), name='char_inputs')
    decoder_inputs = Input(shape=(dec_seq_len, n_mels), name='decoder_inputs')
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
                      n_mels=n_mels,
                      output_per_step=output_per_step,
                      name='decoder')
    post_processing = PostProcessing(hidden_size=hidden_size // 2,
                                     conv1d_bank_depth=post_conv1_bank_depth,
                                     convprojec_filters1=post_convprojec_filters1,
                                     convprojec_filters2=post_convprojec_filters2,
                                     highway_depth=post_highway_depth,
                                     n_fft=target_size,
                                     name='postprocessing')

    char_enc = char_encoder(char_inputs)
    spk_embed = speaker_encoder(spk_inputs)
    conditioned_char_enc = condition([char_enc, spk_embed])
    decoder_pred, alignments = decoder([conditioned_char_enc, decoder_inputs], initial_state=None)
    postnet_out = post_processing(decoder_pred)

    full_model = Model(inputs=[char_inputs, spk_inputs, decoder_inputs],
                       outputs=[decoder_pred, postnet_out, alignments, spk_embed])
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
                          attention_dim=hparams.ATTENTION_DIM,
                          target_size=hparams.TARGET_MAG_FRAME_SIZE,
                          n_mels=hparams.SYNTHESIZER_N_MELS,
                          output_per_step=hparams.OUTPUT_PER_STEP,
                          learning_rate=hparams.LEARNING_RATE,
                          clipnorm=hparams.CLIPNORM,
                          enc_seq_len=None,
                          dec_seq_len=None):
    char_inputs = Input(shape=(enc_seq_len,), name='char_inputs')
    decoder_inputs = Input(shape=(dec_seq_len, n_mels), name='decoder_inputs')
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
                      n_mels=n_mels,
                      output_per_step=output_per_step,
                      name='decoder')
    post_processing = PostProcessing(hidden_size=hidden_size // 2,
                                     conv1d_bank_depth=post_conv1_bank_depth,
                                     convprojec_filters1=post_convprojec_filters1,
                                     convprojec_filters2=post_convprojec_filters2,
                                     highway_depth=post_highway_depth,
                                     n_fft=target_size,
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


def get_SV_test_model(embedded_input=True,
                      pretrained_model=None,
                      n_mels=hparams.SPK_EMBED_N_MELS,
                      lstm_units=hparams.SPK_EMBED_LSTM_UNITS,
                      proj_size=hparams.SPK_EMBED_SIZE,
                      num_layers=hparams.SPK_EMBED_NUM_LAYERS,
                      sliding_window_size=hparams.SLIDING_WINDOW_SIZE):
    pair1 = Input(shape=(proj_size,) if embedded_input else (None, sliding_window_size, n_mels))
    pair2 = Input(shape=(proj_size,) if embedded_input else (None, sliding_window_size, n_mels))
    if not embedded_input:
        pair1_embed, pair2_embed = TestSpeakerEmbedding(lstm_units, proj_size,
                                                        num_layers, name='embeddings')([pair1, pair2])
    else:
        pair1_embed, pair2_embed = pair1, pair2
    sim = TestSpeakerSimilarity(name='similarity')([pair1_embed, pair2_embed])

    model = Model([pair1, pair2], sim)

    if not embedded_input:
        model.load_weights(pretrained_model, by_name=True)

    return model


def load_saved_model(model_path):
    return load_model(model_path, custom_objects=custom_layers)
