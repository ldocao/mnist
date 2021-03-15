import matplotlib.pyplot as plt

class Pixel:
    def __init__(self, pixels):
        self.pixels = pixels

    def display(self):
        image = self.pixels.reshape(28, 28)
        plt.imshow(image, cmap="Greys")
        plt.show()