import keras
import tensorflow as tf

@keras.saving.register_keras_serializable(package="custom")
def conv_to_rnn(inp):
    shape = tf.shape(inp)
    b, h, w, c = shape[0], shape[1], shape[2], shape[3]
    x = tf.transpose(inp, perm=[0, 2, 1, 3])
    x = tf.reshape(x, [b, w, h * c])
    return x

m = keras.models.load_model("captcha_model_best.keras", compile=False)
m.save("portable_model.keras", save_format="keras")

