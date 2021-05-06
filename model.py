import tensorflow as tf

def get_segmentation_model():
    inputs = tf.keras.Input((None, None, 3))

    c1 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
    c1 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2)) (c1)

    c2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2)) (c2)

    c3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2)) (c3)

    c4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (p4)
    c5 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c5)
    p5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2)) (c5)

    c55 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same') (p5)
    c55 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same') (c55)

    u6 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c55)
    u6 = tf.keras.layers.concatenate([u6, c5])
    c6 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
    c6 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

    u71 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u71 = tf.keras.layers.concatenate([u71, c4])
    c71 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (u71)
    c61 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c71)

    u7 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c61)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

    u8 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
    c8 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

    u9 = tf.keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
    c9 = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    # model.summary()

    model.load_weights('checkpoint/rice_leaf_seg.ckpt')
    return model

def get_detection_model():
    model = tf.keras.models.load_model("best_model.h5")
    model.load_weights("best_model.h5")
    # model.summary()
    return model