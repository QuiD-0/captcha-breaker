import numpy as np
import tensorflow as tf
from glob import glob
from tensorflow import keras
from keras import layers
from keras.models import load_model
from PIL import Image

max_length = 6
characters = sorted({'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y' ,'Z'})

# Encode Labels
char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary=list(characters), num_oov_indices=0, mask_token=None
)

num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), num_oov_indices=0, mask_token=None, invert=True
)

lodedmodel = load_model('./model/model_save_test')

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode('utf-8')
        output_text.append(res)
    return output_text

def predit(cap_img) :
    img = tf.io.read_file(glob(cap_img)[0])
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [80, 280])
    img = tf.transpose(img, perm=[1, 0, 2])
    target = tf.expand_dims(img, 0)

    preds = lodedmodel.predict(target)
    return str(decode_batch_predictions(preds)[0])