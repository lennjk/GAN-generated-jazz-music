# GAN-generated-jazz-music

This projects purpose was to build a custom GAN architecture using convolutional NNs to generate jazz music. The principal Idea was to treat music as an image. Like piano covers on Youtube always visualize music as a flow of key presses moving into the piano wherever the player presses the keys, I plotted key presses in a 2d array where each element is a list of length 88 for the keys on the piano. This allows me to generate the music piece as a whole, like an image and then visualize the learning process. For more details I recommend you to check out my power point presentation on the matter.

The jazz pieces were downloaded from kaggle.com and numbered around 1000.

The code is seperated in 4 Files. One that handles midi to array transformations and back, one that handles preprocessing, one that handles model training and finally one that handels predictions.
