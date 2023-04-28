import base64
from dataset import MNIST_Dataset
from flask import Flask, flash, render_template, request
from io import BytesIO
from matplotlib.figure import Figure
import numpy as np
import pickle

app = Flask(__name__)

global y_pred_nn

test_data = MNIST_Dataset("data/fashion_mnist_test.csv")

test_data.load()

test_data.normalize()

model_nn_clf = pickle.load(open('model.pkl', 'rb'))

y_pred_nn = model_nn_clf.predict(test_data.x)

@app.route("/")
def home_page():
    data = []
    return render_template("MNIST_search_engine.html", data=data)

@app.route("/search", methods=["GET", "POST"])
def search():
    global y_pred_nn
    data = []

    for i in range(1000):
        if np.argmax(y_pred_nn[i]) == int(request.form["selector"]):
            probability = y_pred_nn[i][np.argmax(y_pred_nn[i])]
            if probability > .99:

                print("/n")
                print(y_pred_nn[i])
                print(np.argmax(y_pred_nn[i]))
                print(probability)

                image = test_data.get_image_at(i)
                fig = Figure()
                ax = fig.subplots()
                ax.imshow(image, cmap='gray')

                # Save image to a temporary buffer.
                buf = BytesIO()
                fig.savefig(buf, format="png")

                data.append(base64.b64encode(buf.getbuffer()).decode("ascii"))

    return render_template("MNIST_search_engine.html", data=data)

if __name__ == "__main__":
    app.run(host="0.0.0.0")
