The MNIST Character dataset is a traditional AI dataset of 60,000 hand-written characters on a 28x28 grid. This dataset is often used to teach initial AI algorithms. The program below is similar, but uses clothing instead.

Purpose: This program takes a user desired search from a list of ten clothing types. It then uses that information to visually search a database of 20,000 clothing items using nothing but the image.

How: As input, the program breaks the 28x28 image into an array of 784 float variables which are fed into a neural network (in this case with an input, two hidden layers, and an output layer). Using a series of weights, the program takes the input characters and calculates a probability that the image is a shirt, shoes, etc. If the probability or confidence is above a threshold (in this case 99%) then it displays the image in your search area.

Training: To accomplish this, the neural network was trained using a set of 60,000 clothing articles that were already classified. At first the neural network knew nothing, but after the first iteration, it looked at the solution it calculated and the actual solution then backwards propagated that error through the network. Continuing that process over the 60,000 training items, it achieved a 90% accuracy rating using a training set it had not seen yet.

Limitations: The limitation of a simple dense neural network for image detection is that it is heavily dependent on orientation and location within the image. Move the shirt slightly to the left or rotate it and the algorithms will quickly erode. To solve challenges like that you require a more advance algorithm like Convolution Neural Networks. Come back soon for an example implementation of a CNN.

