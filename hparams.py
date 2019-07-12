PADDING_CHAR = '#'
EOS = '$'
# VOCAB = PADDING_CHAR + EOS + ' !"\'(),-./:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyzæéê—'
VOCAB = PADDING_CHAR + EOS + ' !"\',-./?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzæéê—'

BATCH_SIZE = 32
NUM_BUCKETS = 20
MIN_UTT_LEN = 2  # in sec
MAX_UTT_LEN = 15  # In sec

SAMPLE_RATE = 16000
PREEMPHASIZE = 0.97
SYNTHESIZER_N_MELS = 80
OUTPUT_PER_STEP = 5
SPK_EMBED_N_MELS = 40
N_FFT = 2048
WINDOW = 'hann'
HOP_LENGTH = int(0.0125 * SAMPLE_RATE)
WIN_LENGTH = int(0.05 * SAMPLE_RATE)
REF_DB = 20
MAX_DB = 100

CHAR_EMBED_SIZE = 256
SPK_EMBED_SIZE = 256
ENC_CONV1_BANK_DEPTH = 16
ENC_CONVPROJEC_FILTERS1 = 128
ENC_CONVPROJEC_FILTERS2 = 128
ENC_HIGHWAY_DEPTH = 4
HIDDEN_SIZE = 256
POST_CONV1_BANK_DEPTH = 8
POST_CONVPROJEC_FILTERS1 = 256
POST_CONVPROJEC_FILTERS2 = SYNTHESIZER_N_MELS
POST_HIGHWAY_DEPTH = 4
ATTENTION_DIM = 256
TARGET_MAG_FRAME_SIZE = 1 + N_FFT // 2

SLIDING_WINDOW_SIZE = 160
SPK_EMBED_LSTM_UNITS = 512
SPK_EMBED_NUM_LAYERS = 3

LEARNING_RATE = 0.001
WARMUP_STEPS = 4000.0
CLIPNORM = 1.0

vad_window_length = 30  # In milliseconds
vad_moving_average_width = 8
vad_max_silence_length = 6
int16_max = (2 ** 15) - 1
