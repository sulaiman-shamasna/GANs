# GANs
---

*Generative Adversireal Networks - (GANs)* are a special architecture of neural networks that are comprised of two separate neurral networks, i.e., the *Generator* and the *Discriminator*, and the networks are trained through a competitive dynamic. 

The term *generative* indicates the overall purpose of the model, i.e., generating/ creating new data. On the other hand, the term *adverserial* refers to the game-like, competitive dynamic between the two models that constitutes GAn framework. Finally, the term *network* indicates the class of machine learning models most commonly used to represent the Generator and the Discriminator: newral networks.

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



## References
- [GANs in Action](https://www.google.de/books/edition/GANs_in_Action/HojvugEACAAJ?hl=en)