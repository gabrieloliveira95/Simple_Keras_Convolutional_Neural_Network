from tensorflow.keras.datasets import cifar10
from PIL import Image


if __name__ == "__main__":

    imagesAmount = input("How Mutch Images do You Want? ")

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    for i in range(int(imagesAmount)):
        im = Image.fromarray(X_test[i])
        im.save("{}.png".format(i))
