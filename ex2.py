from tensorflow import keras

from ex1 import save_training_plots


def run_ex2():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    x_train = x_train / 255.
    x_test = x_test / 255.
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = keras.Sequential()

    model.add(
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(32, 32, 3), padding='same'))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(rate=0.25))

    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(rate=0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6),
                  metrics='accuracy')

    model.summary()

    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=20, verbose=1,
                        shuffle=True)
    model.save('models/ex2.h5')

    model = keras.models.load_model('models/ex2.h5')
    save_training_plots(history,"ex2")

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Loss:", loss)
    print("Accuracy:", accuracy)


if __name__ == '__main__':
    run_ex2()

