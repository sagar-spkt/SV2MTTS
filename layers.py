import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer, Embedding, Conv1D, BatchNormalization, Activation, Dropout, \
    Bidirectional, RNN, LSTMCell, Dense, StackedRNNCells, LSTM, TimeDistributed


class Conv1DBankStep(Layer):
    def __init__(self, filters, kernel_size, dropout_rate, activation='relu', **kwargs):
        super(Conv1DBankStep, self).__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.activation = activation

        self.conv = Conv1D(filters=self.filters, kernel_size=self.kernel_size,
                           strides=1, padding='same')
        self.batch_normalization = BatchNormalization()
        self.activation_layer = Activation(self.activation)
        self.dropout = Dropout(rate=self.dropout_rate)

    def call(self, inputs, **kwargs):
        inputs = self.conv(inputs)
        inputs = self.batch_normalization(inputs)
        inputs = self.activation_layer(inputs)
        inputs = self.dropout(inputs)

        return inputs

    def get_config(self):
        config = super(Conv1DBankStep, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'activation': self.activation
        })
        return config


class Conv1DBank(Layer):
    def __init__(self, depth, filters, kernel_size, dropout_rate, **kwargs):
        super(Conv1DBank, self).__init__(**kwargs)
        self.depth = depth
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

        for i in range(self.depth):
            setattr(self, 'conv_layer' + str(i), Conv1DBankStep(self.filters, self.kernel_size, self.dropout_rate))

    def call(self, inputs, **kwargs):
        for i in range(self.depth):
            inputs = getattr(self, 'conv_layer' + str(i))(inputs)
        return inputs

    def get_config(self):
        config = super(Conv1DBank, self).get_config()
        config.update({
            'depth': self.depth,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate
        })
        return config


class ZoneoutLSTMCell(LSTMCell):
    def __init__(self, units, zoneout_factor, **kwargs):
        super(ZoneoutLSTMCell, self).__init__(units, **kwargs)

        self.zoneout_factor = zoneout_factor

    def call(self, inputs, states, training=None):
        prev_h, prev_c = states
        output, [new_h, new_c] = super(ZoneoutLSTMCell, self).call(inputs, states, training=training)

        if training is None:
            training = K.learning_phase()

        h, c = tf.cond(training,
                       lambda: ((1 - self.zoneout_factor) * K.dropout(new_h - prev_h, self.zoneout_factor) + prev_h,
                                (1 - self.zoneout_factor) * K.dropout(new_c - prev_c, self.zoneout_factor) + prev_c),
                       lambda: ((1 - self.zoneout_factor) * new_h + self.zoneout_factor * prev_h,
                                (1 - self.zoneout_factor) * new_c + self.zoneout_factor * prev_c))

        return output, [h, c]

    def get_config(self):
        config = super(ZoneoutLSTMCell, self).get_config()
        config.update({
            'zoneout_factor': self.zoneout_factor
        })
        return config


class Prenet(Layer):
    def __init__(self, units, depth, dropout_rate, activation='relu', **kwargs):
        super(Prenet, self).__init__(**kwargs)
        self.units = units
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.activation = activation

        for i in range(self.depth):
            setattr(self, 'prenet_dense' + str(i), Dense(self.units, activation=self.activation))
            setattr(self, 'prenet_dropout' + str(i), Dropout(rate=self.dropout_rate))

    def call(self, inputs, **kwargs):
        for i in range(self.depth):
            inputs = getattr(self, 'prenet_dense' + str(i))(inputs)
            inputs = getattr(self, 'prenet_dropout' + str(i))(inputs, training=True)

        return inputs

    def get_config(self):
        config = super(Prenet, self).get_config()
        config.update({
            'units': self.units,
            'depth': self.depth,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation
        })
        return config


class LocationSensitiveAttention(Layer):
    def __init__(self, attention_dim, filters, kernel_size, cumulate=True, **kwargs):
        super(LocationSensitiveAttention, self).__init__(**kwargs)

        self.attention_dim = attention_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.cumulate = cumulate

        self.query_layer = Dense(self.attention_dim, use_bias=False)
        self.memory_layer = Dense(self.attention_dim, use_bias=False)
        self.location_conv = Conv1D(filters=self.filters, kernel_size=self.kernel_size, padding='same')
        self.location_layer = Dense(self.attention_dim, use_bias=False)
        self.attention_variable = Dense(1, use_bias=False)

    def build(self, input_shape):
        self.attention_bias = self.add_weight('attention_bias', shape=[self.attention_dim],
                                              initializer='glorot_uniform', trainable=True)
        super(LocationSensitiveAttention, self).build(input_shape)

    def call(self, inputs, state=None, **kwargs):
        query, values, keys = inputs
        processed_query = self.query_layer(query)
        processed_query = K.expand_dims(processed_query, axis=1)

        expanded_alignments = K.expand_dims(state, axis=2)
        location_features = self.location_conv(expanded_alignments)
        projected_location_features = self.location_layer(location_features)

        score = self.attention_variable(
            K.tanh(keys + processed_query + projected_location_features + self.attention_bias))  # TODO mask score
        alignment = K.softmax(score, axis=1)

        attention = alignment * values
        attention = K.sum(attention, axis=1)

        alignment = K.squeeze(alignment, axis=2)
        if self.cumulate:
            attention_state = alignment + state
        else:
            attention_state = alignment

        return attention, alignment, attention_state

    def get_config(self):
        config = super(LocationSensitiveAttention, self).get_config()
        config.update({
            'attention_dim': self.attention_dim,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'cumulate': self.cumulate
        })
        return config


class Encoder(Layer):
    def __init__(self,
                 vocab_size,
                 char_embed_size,
                 hidden_size,
                 conv1d_bank_depth,
                 filters,
                 kernel_size,
                 dropout_rate,
                 zoneout_rate,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.vocab_size = vocab_size
        self.char_embed_size = char_embed_size
        self.hidden_size = hidden_size
        self.conv1d_bank_depth = conv1d_bank_depth
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.zoneout_rate = zoneout_rate

        self.embedding = Embedding(self.vocab_size, self.char_embed_size, mask_zero=True)
        self.conv1d_bank = Conv1DBank(self.conv1d_bank_depth, self.filters, self.kernel_size, self.dropout_rate)
        self.bidirectional_lstm = Bidirectional(
            RNN(ZoneoutLSTMCell(self.hidden_size, self.zoneout_rate), return_sequences=True, return_state=False))

    def call(self, inputs, **kwargs):
        inputs = self.embedding(inputs)
        inputs = self.conv1d_bank(inputs)
        inputs = self.bidirectional_lstm(inputs)

        return inputs

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'char_embed_size': self.char_embed_size
        })
        return config


class Decoder(Layer):
    def __init__(self,
                 prenet_units,
                 prenet_depths,
                 dropout_rate,
                 attention_dim,
                 filters,
                 kernel_size,
                 lstm_units,
                 zoneout_factor,
                 n_mels,
                 output_per_step,
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.prenet_units = prenet_units
        self.prenet_depths = prenet_depths
        self.dropout_rate = dropout_rate
        self.attention_dim = attention_dim
        self.filters = filters
        self.kernel_size = kernel_size
        self.lstm_units = lstm_units
        self.zoneout_factor = zoneout_factor
        self.n_mels = n_mels
        self.output_per_step = output_per_step

        self.prenet = Prenet(self.prenet_units, self.prenet_depths, self.dropout_rate)
        self.stack_lstm_cells = StackedRNNCells([ZoneoutLSTMCell(self.lstm_units, self.zoneout_factor)])
        self.attention_mechanism = LocationSensitiveAttention(self.attention_dim, self.filters, self.kernel_size)
        self.frame_projection = Dense(self.n_mels * self.output_per_step)
        self.stop_projection = Dense(self.output_per_step)

    def call(self, inputs, initial_state=None, training=None, **kwargs):
        memory, dec_inputs = inputs

        dec_original_shape = K.shape(dec_inputs)

        dec_inputs_reshaped = K.reshape(dec_inputs, [dec_original_shape[0], -1, self.n_mels * self.output_per_step])

        values = memory  # TODO mask option for memory
        keys = self.attention_mechanism.memory_layer(values)

        if training is None:
            training = K.learning_phase()

        if initial_state is None:
            initial_state = [K.tile(K.expand_dims(K.sum(K.zeros_like(memory), axis=[1, 2])),
                                    [1, self.n_mels * self.output_per_step]),
                             K.sum(K.zeros_like(memory), axis=1),
                             self.stack_lstm_cells.get_initial_state(batch_size=dec_original_shape[0],
                                                                     dtype=dec_inputs.dtype),
                             K.sum(K.zeros_like(memory), axis=2)]

        def step(dec_input, states):
            prev_out, prev_attention, prev_lstm_state, prev_attention_state = states

            dec_input = tf.cond(training, lambda: dec_input, lambda: prev_out)

            prenet_out = self.prenet(dec_input)
            lstm_inputs = K.concatenate([prenet_out, prev_attention], axis=-1)
            lstm_outputs, next_lstm_state = self.stack_lstm_cells(lstm_inputs, states=prev_lstm_state)
            context_vector, alignment, next_attention_state = self.attention_mechanism([lstm_outputs, values, keys],
                                                                                       state=prev_attention_state)
            projections_input = K.concatenate([lstm_outputs, context_vector], axis=-1)
            cell_output = self.frame_projection(projections_input)
            stop_token = self.stop_projection(projections_input)

            return [cell_output, stop_token, alignment], [cell_output, context_vector, next_lstm_state,
                                                          next_attention_state]

        _, all_outputs, _ = K.rnn(step, dec_inputs_reshaped, initial_state)

        dec_outputs = K.reshape(all_outputs[0], (dec_original_shape[0], dec_original_shape[1], self.n_mels))
        stop_tokens = K.reshape(all_outputs[1], (dec_original_shape[0], dec_original_shape[1]))
        alignments = all_outputs[2]

        return dec_outputs, stop_tokens, alignments

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            'prenet_units': self.prenet_units,
            'prenet_depths': self.prenet_depths,
            'dropout_rate': self.dropout_rate,
            'attention_dim': self.attention_dim,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'lstm_units': self.lstm_units,
            'zoneout_factor': self.zoneout_factor,
            'n_mels': self.n_mels,
            'output_per_step': self.output_per_step
        })
        return config


class Postnet(Layer):
    def __init__(self, depth, filters, kernel_size, dropout, activation='tanh', residual_connect=True, **kwargs):
        super(Postnet, self).__init__(**kwargs)

        self.depth = depth
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.activation = activation
        self.residual_connect = residual_connect

        for i in range(self.depth - 1):
            setattr(self, 'postnet_conv' + str(i),
                    Conv1DBankStep(self.filters, self.kernel_size, dropout_rate=self.dropout,
                                   activation=self.activation))
        setattr(self, 'postnet_conv' + str(self.depth - 1),
                Conv1DBankStep(self.filters, kernel_size, self.dropout, activation='linear'))

    def build(self, input_shape):
        if self.residual_connect:
            self.residual_projection = Dense(input_shape[-1])
        super(Postnet, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        for i in range(self.depth):
            x = getattr(self, 'postnet_conv' + str(i))(x)

        if self.residual_connect:
            x = self.residual_projection(x)
            x = x + inputs
        return x

    def get_config(self):
        config = super(Postnet, self).get_config()
        config.update({
            'depth': self.depth,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout,
            'activation': self.activation,
            'residual_connect': self.residual_connect
        })
        return config


class Conditioning(Layer):
    def call(self, inputs, **kwargs):
        memory, speaker_embedding = inputs
        tiled_speaker_embeddings = K.tile(K.expand_dims(speaker_embedding, axis=1), [1, K.shape(memory)[1], 1])
        conditioned_memory = K.concatenate([memory, tiled_speaker_embeddings], axis=-1)
        return conditioned_memory


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


custom_layers = {
    LocationSensitiveAttention.__name__: LocationSensitiveAttention,
    ZoneoutLSTMCell.__name__: ZoneoutLSTMCell,
    Decoder.__name__: Decoder,
    Prenet.__name__: Prenet,
    Conv1DBankStep.__name__: Conv1DBankStep,
    Conv1DBank.__name__: Conv1DBank,
    Encoder.__name__: Encoder,
    Conditioning.__name__: Conditioning,
    Postnet.__name__: Postnet,
    TrainSpeakerEmbedding.__name__: TrainSpeakerEmbedding,
    InferenceSpeakerEmbedding.__name__: InferenceSpeakerEmbedding
}
