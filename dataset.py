import pandas as pd
import matplotlib.pyplot as plt


class MNIST_Dataset:
    def __init__(self, file_location=""):
        self.file_location = file_location
        self.data_file = pd.DataFrame()
        self.x = pd.DataFrame()
        self.y = pd.DataFrame()

    def load(self, os_loc=None):
        """
        Loads a data set for use in the machine learning program.

        :param os_loc: [Optional] String representing the file to be loaded. If None, the function
        will request user provide a location via the command prompt. Default = None.

        :return: DataFrame representing the file loaded.
        """

        if os_loc is None:
            self.data_file = pd.read_csv(self.file_location)

        else:
            self.file_location=os_loc
            self.data_file = pd.read_csv(self.file_location)

        self.y = self.data_file["label"]
        self.x = self.data_file.drop(["label"], axis=1)
        print(self.x.describe())
        print(self.y.describe())
        return self.data_file

    def get_image_at(self, position=0):
        """
        Displays an image representing the character following processing.

        :param position : Position of the image in the array to be plotted.
        :return: 28x28 image representing the data at that location.
        """

        image = self.x.iloc[position]
        image = image.to_numpy().reshape((28, 28))
        return image

    def normalize(self):
        self.x = self.x.apply(lambda x: x/255)