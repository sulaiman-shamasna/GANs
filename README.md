# GANs
---

*Generative Adversireal Networks - (GANs)* are a special architecture of neural networks that are comprised of two separate neurral networks, i.e., the *Generator* and the *Discriminator*, and the networks are trained through a competitive dynamic.

The term *generative* indicates the overall purpose of the model, i.e., generating/ creating new data. On the other hand, the term *adverserial* refers to the game-like, competitive dynamic between the two models that constitutes GAN framework: The Generator and the Discriminator. Finally, the term *network* indicates the class of machine learning models most commonly used to represent the Generator and the Discriminator: newral networks.

## How do GANs work?
The Generator's purpose is trying to generate new data based on the training dataset, where as the Discriminator is supposed to distinguish these artificially generated data from the real ones.

The Generator learns through the feedback it receives from the Discriminator’s classifications. The Discriminator’s goal is to determine whether a particular example is real (coming from the training dataset) or fake (created by the Generator). Accordingly, each time the Discriminator is fooled into classifying a fake image as real, the Generator knows it did something well. Conversely, each time the Discriminator correctly rejects a Generator-produced image as fake, the Generator receives the feedback that it needs to improve. The two networks keep fighting/ improving until they reach an optimal point, i.e., *Nash Equilibrium*; a state of which any movement from any of the two sides will not change the strategy of the game.

|   **X**   | **Generator** | **Discriminator** |
|-----------|---------------|-------------------|
|**Input**  | A vector of random numbers | The Discriminator receives input from two sources: (1) Real examples coming from the training dataset (2) Fake examples coming from the Generator
|**Output** | Fake examples that strive to be as convincing as possible | Predicted probability that the input example is real
|**Goal**   | Generate fake data that is indistinguishable from member of the trainingdataset | Distinguish between the fake examples coming from the Generator and the real examples coming from the training dataset


## Training GANs

The Figure bellow illustrates tow fundamintal steps in training *GANs*.

![trainin_gans2.PNG](https://github.com/sulaiman-shamasna/GANs/blob/main/plots/gen_disc_train2.svg)


1. ***Training the Discriminator***.
    - Take a random real example *X* from the training dataset.
    - Get a new random noise vector *Z* and, using the Generator network, synthesize a fake example *X´*.
    - Use the Discriminator network to classify *X* and *X´*.
    - Compute the classification errors and backpropagate the total error to update the Discriminator weights and biases, seeking to *minimize* the classification errors.
2. ***Training the Generator***.
    - Get a new random noise vector *Z* and, using the Generator network, synthesize a fake example *X´*.
    - Use the Discriminator network to classify *X´*.
    - Compute the classification error and backpropagate the error to update the Generator weights and biases, seeking to *maximize* the Discriminator's error.

## Reaching Equilibrium
In a GAN, the two networks have competing objectives: when one network gets better, th other gets worse. How do we determine when to stop?

From Game Theory point of view, this step may be recognized as a *zero-sum game*—a situation in which one player’s gains equal the other player’s losses. When one player improves by a certain amount, the other player worsens by the same amount. All *zero-sum games* have a *Nash equilibrium*, a point at which neither player can improve their situation or payoff by changing their actions.

GAN reaches Nash equilibrium when the following conditions are met:
- The Generator produces fake examples that are indistinguishable from the real data in the training dataset.
- The Discriminator can at best randomly guess whether a particular example is real or fake (that is, make a 50/50 guess whether an example is real).

**NOTE** Nash equilibrium is named after the American economist and mathematician *John Forbes Nash Jr.*, whose life story and career were captured in the biography titled A *Beautiful Mind* and inspired the eponymous film.

## **Autoencoders**

### Autoencoder Structure

By looking at the structure of an autoencoder, images are used as an example, this structure, however, aoolies in other cases, e.g., language, etc. As of any other advancement in machine learning, the high-level idea of autoencoders is intuitive and follows three main steps illustrated in the figure bellow.

**DEFINISION** The *latent space* is the hidden representation of the data. Rather than expressing words or images in their uncompressed versions, an autoencoder compresses and clusters them based on its understanding of the data.

![Autoencoder](https://github.com/sulaiman-shamasna/GANs/blob/main/plots/autoencoder2.svg)


1. ***Encoder network***: A representation *X* is taken then the dimension is reduced from *y* to *z* by using learnt encoder (typically, a one-or many-layer neural network).
2. ***Latent space (z)***: As training goes on, the latent space is being established to have some meaning. Latent space is typically a representation of a smaller dimension and acts as an intermediate step. In this representation of this data, the autoencoder is trying to “organize its thoughts”.
3. ***Decoder network***: By using the decoder, the original object is reconstructed into the original dimension. This is typically done by a neural network that is a
mirror image of the encoder, which is the step from z to x*. A reversed process is applied of the encoding to get back, e.g., a 784 pixel-values long recon-
structed vector (of a 28 × 28 image) from the 256 pixel-values long vector of the
latent space.

### Example of Autoencoder Training
1. We take images *x* and feed them through the autoencoder.
2. We get out *x´*, reconstruction of the images.
3. We measure the reconstruction loss—the difference between *x* and *x´*.
    -   This is done using a distance (for example, mean average error) between the pixels of *x* and *x´*.
    - This gives us an explicit objective function - *reconstruction loss* (*|| *x* –*x´* ||*) to optimize via a version of gradient descent.

So we are trying to find the parameters of the encoder and the decoder that would
minimize the reconstruction loss that we update by using gradient descent

### Variational Autoencoders - (VAEs)
The primary difference between a *Variational Autoencoder (VAE)* and a normal autoencoder lies in the representation of the latent space. A *VAE* models the latent space as a distribution characterized by a learned mean and standard deviation, typically using a *multivariate Gaussian distribution*. This approach is rooted in Bayesian machine learning, where the goal is to learn the parameters of this distribution rather than simply an array of numbers.

In practice, this means VAEs sample from the learned latent distribution, generating values that are then processed by the decoder to produce new examples that resemble the original dataset.


### Kullback–Leibler divergence
In mathematical statistics, the ***Kullback–Leibler (KL) divergence*** (also called ***relative entropy*** and ***I-divergence***), is a type of statistical distance: a measure of how one reference probability distribution ***P*** is different from a second probability distribution ***Q***. Mathematically, it is defined as


<p align="center">
  <img src="https://latex.codecogs.com/png.latex?D_{\text{KL}}(P\parallel%20Q)%20=%20\sum_{x%20\in%20\xi}%20P(x)%20\log\left(\frac{P(x)}{Q(x)}\right)" alt="KL Divergence Equation">
</p>

## References
- [GANs in Action](https://www.google.de/books/edition/GANs_in_Action/HojvugEACAAJ?hl=en)