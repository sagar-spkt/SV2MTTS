from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input

from layers import Encoder, Decoder, PostProcessing, Conditioning, InferenceSpeakerEmbedding


def get_train_model(vocab_size,
                    char_embed_size,
                    sliding_window_size,
                    spk_embed_lstm_units,
                    spk_embed_size,
                    spk_embed_num_layers,
                    enc_conv1_bank_depth,
                    enc_convprojec_filters1,
                    enc_convprojec_filters2,
                    enc_highway_depth,
                    hidden_size,
                    post_conv1_bank_depth,
                    post_convprojec_filters1,
                    post_convprojec_filters2,
                    post_highway_depth,
                    dec_frsize,
                    target_size,
                    n_mels,
                    embed_mels,
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
    decoder_pred, *latest_states = decoder([conditioned_char_enc, decoder_inputs], initial_state=None)
    postnet_out = post_processing(decoder_pred)

    train_model = Model(inputs=[char_inputs, spk_inputs, decoder_inputs],
                        outputs=[decoder_pred, postnet_out])
    return train_model


def get_speaker_embedding_inference_model(pretrained_model,
                                          n_mels,
                                          lstm_units,
                                          proj_size,
                                          num_layers,
                                          sliding_window_size):
    inputs = Input(shape=(None, sliding_window_size, n_mels))
    emb = InferenceSpeakerEmbedding(lstm_units, proj_size, num_layers, name='embeddings')(inputs)
    model = Model(inputs, emb)

    model.load_weights(pretrained_model, by_name=True)

    return model
