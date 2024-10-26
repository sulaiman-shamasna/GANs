# GANs

![trainin_gans2.PNG](https://github.com/sulaiman-shamasna/GANs/blob/main/plots/gen_disc_train2.svg)

The Figure above illustrates tow fundamintal steps in training *GANs*, they are the following:
1. **Training the Discriminator**.
    - Take a random real example *X* from the training dataset.
    - Get a new random noise vector *Z* and, using the Generator network, synthesize a fake example *X´*.
    - Use the Discriminator network to classify *X* and *X´*.
    - Compute the classification errors and backpropagate the total error to update the Discriminator weights and biases, seeking to *minimize* the classification errors.
2. **Training the Generator**.
    - Get a new random noise vector *Z* and, using the Generator network, synthesize a fake example *X´*.
    - Use the Discriminator network to classify *X´*.
    - Compute the classification error and backpropagate the error to update the Generator weights and biases, seeking to *maximize* the Discriminator's error.