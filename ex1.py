import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras

def save_training_plots(history, model_name):
    #  "Accuracy"
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'plots/accuracy_{model_name}.png')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'plots/loss_{model_name}.png')
    plt.show()


def run_ex1_b():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape((60000, 28, 28, 1)) / 255.
    x_test = x_test.reshape((10000, 28, 28, 1)) / 255.
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(rate=0.25))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adadelta(learning_rate=1),
                  metrics='accuracy')

    model.summary()

    history = model.fit(x_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)
    model.save('models/ex1_b.h5')

    model = keras.models.load_model('models/ex1_b.h5')
    save_training_plots(history,"ex1_b_with_droput")

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Loss:", loss)
    print("Accuracy:", accuracy)


if __name__ == '__main__':
    # run_training()
    run_ex1_b()

