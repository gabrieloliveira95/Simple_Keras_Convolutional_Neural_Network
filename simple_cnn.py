from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import model_from_json
from tensorflow.keras.datasets import cifar10


def model():
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=3,
                     padding='same', activation='relu', input_shape=[32, 32, 3]))
    model.add(Conv2D(
        filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool2D(
        pool_size=2, strides=2, padding='valid',))
    model.add(Conv2D(
        filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(
        filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPool2D(
        pool_size=2, strides=2, padding='valid'))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=10, activation='softmax'))
    return model


def loadModelAndWeights():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # Evaluate loaded model on test data
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='Adam', metrics=['sparse_categorical_crossentropy'])
    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    print(f'Loss = {test_loss}')
    print(f'Accuracy = {test_accuracy}')


if __name__ == "__main__":

    class_names = ['airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train / 255.0  # Values 0-255 to 0-1
    x_test = x_test / 255.0    # Values 0-255 to 0-1

    print(f'Shape: {x_train.shape}')

    model = model()

    print(model.summary())

    print('\n\nDo You Want To Train This Model??')
    confirm = input(
        "It may take a long time... [y or n]: ")

    if confirm.upper() == 'Y':
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='Adam', metrics=['sparse_categorical_crossentropy'])

        model.fit(x_train, y_train, epochs=5)

        test_loss, test_accuracy = model.evaluate(x_test, y_test)

        print(f'Loss = {test_loss}')
        print(f'Accuracy = {test_accuracy}')

        save = input(
            "Do You Want to Save This Model and Weights? [y or n]: ")
        if save.upper() == 'Y':
            # Model to JSON
            model_json = model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(model_json)
            # Weights to HDF5
            model.save_weights("model.h5")
            print("Saved model to disk")

            testSavedModel = input(
                "You want to Test The Saved Models and Weights? [y or n]: ")
            if testSavedModel.upper() == 'Y':
                loadModelAndWeights()
