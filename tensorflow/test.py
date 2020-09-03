import tensorflow as tf
tf.debugging.set_log_device_placement(True)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(gpus)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(logical_gpus)
print(str(tf.__version__))
