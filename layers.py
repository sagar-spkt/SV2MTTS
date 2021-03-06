import tensorflow as tf
from tensorflow.python import Constant
from tensorflow.python.keras import backend as K, initializers
from tensorflow.python.keras.constraints import NonNeg
from tensorflow.python.keras.layers import Layer, Dense, Embedding, Bidirectional, GRU, Add, Dropout, MaxPooling1D, \
    Conv1D, BatchNormalization, Activation, Lambda, Multiply, Reshape, GRUCell, LSTM, TimeDistributed


class BahdanauAttention(Layer):
    def __init__(self, attention_dim, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.attention_dim = attention_dim

        self.memory_layer = Dense(self.attention_dim, use_bias=False)
        self.query_layer = Dense(self.attention_dim, use_bias=False)
        self.attention_variable = Dense(1)

    def call(self, inputs, **kwargs):
        query, values, keys = inputs

        hidden_with_time_axis = K.expand_dims(query, 1)
        score = self.attention_variable(K.tanh(
            keys + self.query_layer(hidden_with_time_axis)))  # TODO Mask option for score with infinity
        alignment = K.softmax(score, axis=1)
        attention = alignment * values
        alignment = K.squeeze(alignment, axis=2)
        attention = K.sum(attention, axis=1)

        return attention, alignment

    def get_config(self):
        config = super(BahdanauAttention, self).get_config()
        config.update({
            'attention_dim': self.attention_dim
        })
        return config


class Decoder(Layer):
    def __init__(self, hidden_size, attention_dim, n_mels, output_per_step, **kwargs):
        self.hidden_size = hidden_size
        self.attention_dim = attention_dim
        self.n_mels = n_mels
        self.output_per_step = output_per_step

        self.prenet = Prenet()
        self.attn_rnn_cell = GRUCell(self.hidden_size)
        self.attention_mechanism = BahdanauAttention(self.attention_dim)
        self.projection = Dense(self.hidden_size)
        self.decoderRNNCell1 = GRUCell(self.hidden_size)
        self.decoderRNNCell2 = GRUCell(self.hidden_size)
        self.output_projection = Dense(self.n_mels * self.output_per_step)
        super(Decoder, self).__init__(**kwargs)

    def call(self, inputs, initial_state=None, training=None, **kwargs):
        memory, dec_inputs = inputs

        dec_original_shape = K.shape(dec_inputs)

        dec_inputs_reshaped = K.reshape(dec_inputs, [dec_original_shape[0], -1, self.n_mels * self.output_per_step])
        go_frame = K.zeros_like(dec_inputs_reshaped[:, 1, :])
        dec_inputs_with_goframe = K.concatenate([K.expand_dims(go_frame, axis=1), dec_inputs_reshaped[:, :-1, :]],
                                                axis=1)

        values = memory  # TODO mask option for memory
        keys = self.attention_mechanism.memory_layer(memory)

        if training is None:
            training = K.learning_phase()

        if initial_state is None:
            initial_state = [go_frame,
                             K.sum(K.zeros_like(memory), axis=1),
                             K.sum(K.zeros_like(memory), axis=2),
                             self.attn_rnn_cell.get_initial_state(batch_size=dec_original_shape[0],
                                                                  dtype=dec_inputs.dtype),
                             self.decoderRNNCell1.get_initial_state(batch_size=dec_original_shape[0],
                                                                    dtype=dec_inputs.dtype),
                             self.decoderRNNCell2.get_initial_state(batch_size=dec_original_shape[0],
                                                                    dtype=dec_inputs.dtype)]

        def step(dec_input, states):
            (prev_output, prev_attention,
             prev_alignment, prev_attn_rnn_state,
             prev_dec_rnn1_state, prev_dec_rnn2_state) = states

            dec_input = K.switch(training, dec_input, prev_output)

            prenet_out = self.prenet(dec_input)
            cell_inputs = K.concatenate([prenet_out, prev_attention], axis=-1)
            cell_out, next_attn_rnn_state = self.attn_rnn_cell(cell_inputs, [prev_attn_rnn_state])
            next_attention, next_alignment = self.attention_mechanism([cell_out, values, keys])
            concatenated = K.concatenate([next_attention, cell_out], axis=-1)
            projected = self.projection(concatenated)
            dec_rnn1_out, next_dec_rnn1_state = self.decoderRNNCell1(projected, [prev_dec_rnn1_state])
            res_conn1 = projected + dec_rnn1_out
            dec_rnn2_out, next_dec_rnn2_state = self.decoderRNNCell2(res_conn1, [prev_dec_rnn2_state])
            res_conn2 = res_conn1 + dec_rnn2_out
            next_output = self.output_projection(res_conn2)

            return [next_output, next_alignment], [
                next_output, next_attention,
                next_alignment, next_attn_rnn_state,
                next_dec_rnn1_state, next_dec_rnn2_state
            ]

        _, all_outputs, _ = K.rnn(step, dec_inputs_with_goframe, initial_state)
        dec_outputs = K.reshape(all_outputs[0], (dec_original_shape[0], dec_original_shape[1], self.n_mels))
        alignments = all_outputs[1]

        return dec_outputs, alignments

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'attention_dim': self.attention_dim,
            'n_mels': self.n_mels,
            'output_per_step': self.output_per_step
        })
        return config


class Conditioning(Layer):
    def call(self, inputs, **kwargs):
        memory, speaker_embedding = inputs
        tiled_speaker_embeddings = K.tile(K.expand_dims(speaker_embedding, axis=1), [1, K.shape(memory)[1], 1])
        conditioned_memory = K.concatenate([memory, tiled_speaker_embeddings], axis=-1)
        return conditioned_memory


class Prenet(Layer):
    def __init__(self, **kwargs):
        super(Prenet, self).__init__(**kwargs)
        self.FC1 = Dense(256, activation='relu')
        self.dropout1 = Dropout(0.5)
        self.FC2 = Dense(128, activation='relu')
        self.dropout2 = Dropout(0.5)

    def call(self, inputs, **kwargs):
        inputs = self.FC1(inputs)
        inputs = self.dropout1(inputs)
        inputs = self.FC2(inputs)
        inputs = self.dropout2(inputs)

        return inputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        return input_shape[:-1].concatenate(128)


class Conv1DBankStep(Layer):
    def __init__(self, kernel_size, **kwargs):
        super(Conv1DBankStep, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = Conv1D(filters=128, kernel_size=kernel_size,
                           strides=1, padding='same')
        self.batch_normalization = BatchNormalization()
        self.activation = Activation('relu')

    def call(self, inputs, **kwargs):
        inputs = self.conv(inputs)
        inputs = self.batch_normalization(inputs)
        inputs = self.activation(inputs)

        return inputs

    def get_config(self):
        config = super(Conv1DBankStep, self).get_config()
        config.update({
            'kernel_size': self.kernel_size
        })
        return config


class Conv1DBank(Layer):
    def __init__(self, depth, **kwargs):
        super(Conv1DBank, self).__init__(**kwargs)
        self.depth = depth
        for i in range(1, self.depth + 1):
            setattr(self, 'conv_layer' + str(i), Conv1DBankStep(i))

    def call(self, inputs, **kwargs):
        for i in range(1, self.depth + 1):
            inputs = getattr(self, 'conv_layer' + str(i))(inputs)
        return inputs

    def get_config(self):
        config = super(Conv1DBank, self).get_config()
        config.update({
            'depth': self.depth
        })
        return config


class HighwayNetStep(Layer):
    def __init__(self, bias=-3, **kwargs):
        super(HighwayNetStep, self).__init__(**kwargs)
        self.bias = initializers.Constant(value=bias)

        self.multiply1 = Multiply()
        self.multiply2 = Multiply()
        self.add = Add()

    def build(self, input_shape):
        self.T = Dense(units=input_shape[-1],
                       activation='sigmoid',
                       bias_initializer=self.bias)
        self.H = Dense(units=input_shape[-1],
                       activation='relu')
        self.cary_gate = Lambda(lambda x: 1.0 - x,
                                output_shape=(input_shape[-1],))

    def call(self, inputs, **kwargs):
        h = self.H(inputs)
        t = self.T(inputs)
        c = self.cary_gate(t)
        highway_out = self.add([
            self.multiply1([h, t]),
            self.multiply2([inputs, c])
        ])

        return highway_out

    def get_config(self):
        config = super(HighwayNetStep, self).get_config()
        config.update({
            'bias': self.bias
        })
        return config


class HighwayNet(Layer):
    def __init__(self, n_layers, **kwargs):
        super(HighwayNet, self).__init__(**kwargs)
        self.n_layers = n_layers
        for i in range(self.n_layers):
            setattr(self, 'highway_layer' + str(i), HighwayNetStep())

    def call(self, inputs, **kwargs):
        for i in range(self.n_layers):
            inputs = getattr(self, 'highway_layer' + str(i))(inputs)
        return inputs

    def get_config(self):
        config = super(HighwayNet, self).get_config()
        config.update({
            'n_layers': self.n_layers
        })
        return config


class CBHG(Layer):
    def __init__(self,
                 hidden_size,
                 conv1d_bank_depth,
                 convprojec_filters1,
                 convprojec_filters2,
                 highway_depth,
                 return_state,
                 encoder_side=True,
                 **kwargs):
        super(CBHG, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.conv1d_bank_depth = conv1d_bank_depth
        self.convprojec_filters1 = convprojec_filters1
        self.convprojec_filters2 = convprojec_filters2
        self.highway_depth = highway_depth
        self.return_state = return_state
        self.encoder_side = encoder_side

        self.conv1d_bank = Conv1DBank(self.conv1d_bank_depth)
        self.maxpooling1d = MaxPooling1D(pool_size=2, strides=1,
                                         padding='same')
        self.conv1d_projection1 = Conv1D(filters=self.convprojec_filters1, kernel_size=3,
                                         strides=1, padding='same')
        self.bn_projection1 = BatchNormalization()
        self.activation_projection = Activation('relu')
        self.conv1d_projection2 = Conv1D(filters=self.convprojec_filters2, kernel_size=3,
                                         strides=1, padding='same')
        self.bn_projection2 = BatchNormalization()
        self.residual = Add()
        if not self.encoder_side:
            self.affine_transform = Dense(128)
        self.highway_net = HighwayNet(self.highway_depth)
        self.bidirectional_gru = Bidirectional(GRU(self.hidden_size,
                                                   return_sequences=True,
                                                   return_state=self.return_state))

    def call(self, inputs, **kwargs):
        x = self.conv1d_bank(inputs)
        x = self.maxpooling1d(x)
        x = self.conv1d_projection1(x)
        x = self.bn_projection1(x)
        x = self.activation_projection(x)
        x = self.conv1d_projection2(x)
        x = self.bn_projection2(x)
        x = self.residual([inputs, x])
        if not self.encoder_side:
            x = self.affine_transform(x)
        x = self.highway_net(x)
        x = self.bidirectional_gru(x)
        return x

    def get_config(self):
        config = super(CBHG, self).get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'conv1d_bank_depth': self.conv1d_bank_depth,
            'convprojec_filters1': self.convprojec_filters1,
            'convprojec_filters2': self.convprojec_filters2,
            'highway_depth': self.highway_depth,
            'return_state': self.return_state,
            'encoder_side': self.encoder_side
        })
        return config


class Encoder(Layer):
    def __init__(self,
                 hidden_size,
                 vocab_size,
                 embedding_size,
                 conv1d_bank_depth,
                 convprojec_filters1,
                 convprojec_filters2,
                 highway_depth,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.conv1d_bank_depth = conv1d_bank_depth
        self.convprojec_filters1 = convprojec_filters1
        self.convprojec_filters2 = convprojec_filters2
        self.highway_depth = highway_depth

        self.embedding = Embedding(self.vocab_size, self.embedding_size, name='embedding_layer')
        self.enc_prenet = Prenet(name='enc_prenet')
        self.encoder_cbhg = CBHG(self.hidden_size,
                                 self.conv1d_bank_depth,
                                 self.convprojec_filters1,
                                 self.convprojec_filters2,
                                 self.highway_depth,
                                 return_state=False,
                                 name='encoder_cbhg')

    def call(self, inputs, training=None, mask=None):
        x = self.embedding(inputs)
        x = self.enc_prenet(x)
        enc_out = self.encoder_cbhg(x)
        return enc_out

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'vocab_size': self.vocab_size,
            'embedding_size': self.embedding_size,
            'conv1d_bank_depth': self.conv1d_bank_depth,
            'convprojec_filters1': self.convprojec_filters1,
            'convprojec_filters2': self.convprojec_filters2,
            'highway_depth': self.highway_depth
        })
        return config


class PostProcessing(Layer):
    def __init__(self, hidden_size,
                 conv1d_bank_depth, convprojec_filters1,
                 convprojec_filters2, highway_depth,
                 n_fft, **kwargs):
        super(PostProcessing, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.conv1d_bank_depth = conv1d_bank_depth
        self.convprojec_filters1 = convprojec_filters1
        self.convprojec_filters2 = convprojec_filters2
        self.highway_depth = highway_depth
        self.n_fft = n_fft

        self.decoder_cbhg = CBHG(self.hidden_size,
                                 self.conv1d_bank_depth,
                                 self.convprojec_filters1,
                                 self.convprojec_filters2,
                                 self.highway_depth,
                                 return_state=False,
                                 encoder_side=False,
                                 name='decoder_cbhg')
        self.post_dense = Dense(self.n_fft, name='postnet_dense')

    def call(self, inputs, **kwargs):
        cbhg_out = self.decoder_cbhg(inputs)
        post_net_out = self.post_dense(cbhg_out)

        return post_net_out

    def get_config(self):
        config = super(PostProcessing, self).get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'conv1d_bank_depth': self.conv1d_bank_depth,
            'convprojec_filters1': self.convprojec_filters1,
            'convprojec_filters2': self.convprojec_filters2,
            'highway_depth': self.highway_depth,
            'n_fft': self.n_fft,
        })
        return config


class TrainSpeakerEmbedding(Layer):
    def __init__(self, lstm_units, proj_size, num_layers, **kwargs):
        super(TrainSpeakerEmbedding, self).__init__(**kwargs)

        self.lstm_units = lstm_units
        self.proj_size = proj_size
        self.num_layers = num_layers

        for i in range(1, self.num_layers):
            setattr(self, 'lstm' + str(i), LSTM(self.lstm_units, return_sequences=True, name='lstm' + str(i)))
            setattr(self, 'proj' + str(i), TimeDistributed(Dense(self.proj_size), name='proj' + str(i)))

        setattr(self, 'lstm' + str(self.num_layers),
                LSTM(self.lstm_units, return_sequences=False, name='lstm' + str(self.num_layers)))
        setattr(self, 'proj' + str(self.num_layers), Dense(self.proj_size, name='proj' + str(self.num_layers)))

    def call(self, inputs, **kwargs):
        for i in range(1, self.num_layers + 1):
            inputs = getattr(self, 'lstm' + str(i))(inputs)
            inputs = getattr(self, 'proj' + str(i))(inputs)

        # L2-normalize to get embeddings
        embeddings = K.l2_normalize(inputs, axis=-1)
        return embeddings

    def get_config(self):
        config = super(TrainSpeakerEmbedding, self).get_config()
        config.update({
            'lstm_units': self.lstm_units,
            'proj_size': self.proj_size,
            'num_layers': self.num_layers
        })
        return config


class InferenceSpeakerEmbedding(TrainSpeakerEmbedding):
    def call(self, inputs, **kwargs):
        inputs_shape = K.shape(inputs)

        mask = K.cast(K.squeeze(K.any(K.not_equal(inputs, 0.), axis=(-2, -1), keepdims=True), axis=-1),
                      dtype=inputs.dtype)

        inputs_to_lstm = K.reshape(inputs, (-1, inputs.shape[-2], inputs.shape[-1]))

        inputs_embed = super(InferenceSpeakerEmbedding, self).call(inputs_to_lstm)

        inputs_embed = K.reshape(inputs_embed, (inputs_shape[0], inputs_shape[1], inputs_embed.shape[-1]))

        inputs_embed = inputs_embed * mask

        n = K.sum(mask, axis=1)

        inputs_embed = K.sum(inputs_embed, axis=1) / n

        return inputs_embed


class TrainSpeakerSimilarity(Layer):
    def __init__(self, N, M, **kwargs):
        self.N = N
        self.M = M
        super(TrainSpeakerSimilarity, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name='w',
                                 shape=(1,),
                                 initializer=Constant(value=10.),
                                 constraint=NonNeg())
        self.b = self.add_weight(name='b',
                                 shape=(1,),
                                 initializer=Constant(value=-5.))
        super(TrainSpeakerSimilarity, self).build(input_shape)

    def call(self, inputs, **kwargs):
        embedded_split = K.reshape(inputs, shape=(self.N, self.M, -1))

        center = K.l2_normalize(K.mean(embedded_split, axis=1), axis=-1)
        center_except = K.l2_normalize(
            K.reshape(K.sum(embedded_split, axis=1, keepdims=True) - embedded_split, shape=(self.N * self.M, -1)),
            axis=-1)

        similarity = K.concatenate(
            [K.concatenate([K.sum(center_except[i * self.M:(i + 1) * self.M, :] * embedded_split[j, :, :], axis=1,
                                  keepdims=True) if i == j else K.sum(center[i:(i + 1), :] * embedded_split[j, :, :],
                                                                      axis=1, keepdims=True) for
                            i in range(self.N)], axis=1) for j in range(self.N)], axis=0)

        similarity = self.w * similarity + self.b

        return similarity

    def get_config(self):
        config = super(TrainSpeakerSimilarity, self).get_config()
        config.update({
            'N': self.N,
            'M': self.M
        })
        return config


class TestSpeakerEmbedding(TrainSpeakerEmbedding):
    def call(self, inputs, **kwargs):
        pair1, pair2 = inputs

        pair1_shape, pair2_shape = K.shape(pair1), K.shape(pair2)

        pair1_mask = K.cast(K.squeeze(K.any(K.not_equal(pair1, 0.), axis=(-2, -1), keepdims=True), axis=-1),
                            dtype=pair1.dtype)
        pair2_mask = K.cast(K.squeeze(K.any(K.not_equal(pair2, 0.), axis=(-2, -1), keepdims=True), axis=-1),
                            dtype=pair2.dtype)

        pair1_to_lstm = K.reshape(pair1, (-1, pair1.shape[-2], pair1.shape[-1]))
        pair2_to_lstm = K.reshape(pair2, (-1, pair2.shape[-2], pair2.shape[-1]))

        batch = K.concatenate([pair1_to_lstm, pair2_to_lstm], axis=0)

        embedded = super(TestSpeakerEmbedding, self).call(batch)

        pair1_embed = embedded[:K.shape(pair1_to_lstm)[0]]
        pair2_embed = embedded[K.shape(pair1_to_lstm)[0]:]

        pair1_embed = K.reshape(pair1_embed, (pair1_shape[0], pair1_shape[1], -1))
        pair2_embed = K.reshape(pair2_embed, (pair2_shape[0], pair2_shape[1], -1))

        pair1_embed = pair1_embed * pair1_mask
        pair2_embed = pair2_embed * pair2_mask

        pair1_n = K.sum(pair1_mask, axis=1)
        pair2_n = K.sum(pair2_mask, axis=1)

        pair1_embed = K.sum(pair1_embed, axis=1) / pair1_n
        pair2_embed = K.sum(pair2_embed, axis=1) / pair2_n

        return pair1_embed, pair2_embed


class TestSpeakerSimilarity(TrainSpeakerSimilarity):
    def __init__(self, **kwargs):
        super(TrainSpeakerSimilarity, self).__init__(**kwargs)

    def build(self, input_shape):
        super(TrainSpeakerSimilarity, self).build(input_shape)

    def call(self, inputs, **kwargs):
        pair1_embed, pair2_embed = inputs

        pair1_embed = K.l2_normalize(pair1_embed, axis=-1)
        pair2_embed = K.l2_normalize(pair2_embed, axis=-1)

        sim = K.dot(pair1_embed, K.transpose(pair2_embed))

        sim = tf.linalg.tensor_diag_part(sim)

        return sim


custom_layers = {
    BahdanauAttention.__name__: BahdanauAttention,
    Decoder.__name__: Decoder,
    Conditioning.__name__: Conditioning,
    Prenet.__name__: Prenet,
    Conv1DBankStep.__name__: Conv1DBankStep,
    Conv1DBank.__name__: Conv1DBank,
    HighwayNetStep.__name__: HighwayNetStep,
    HighwayNet.__name__: HighwayNet,
    CBHG.__name__: CBHG,
    Encoder.__name__: Encoder,
    PostProcessing.__name__: PostProcessing,
    TrainSpeakerEmbedding.__name__: TrainSpeakerEmbedding,
    InferenceSpeakerEmbedding.__name__: InferenceSpeakerEmbedding,
    TrainSpeakerSimilarity.__name__: TrainSpeakerSimilarity,
    TestSpeakerEmbedding.__name__: TestSpeakerEmbedding,
    TestSpeakerSimilarity.__name__: TestSpeakerSimilarity
}
