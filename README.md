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

$$
D_{\text{KL}}(P\parallel Q) = \sum_{x \in \xi} P(x) \log \left( \frac{P(x)}{Q(x)} \right)
$$

## **Generative Adversarial Networks - GANs**

The main components of *GANs* are  the Generator and the Discriminator, that are represented by differentiable functions, such as neural networks, each with its own cost function. The two networks are trained by *backpropagation* by using the *Discriminator’s loss*. The Discriminator strives to *minimize* the loss for both the real and the fake examples, while the Generator tries to *maximize* the Discriminator’s loss for the fake examples it produces. This dynamic is summarized in the figure bellow.

![Autoencoder](https://github.com/sulaiman-shamasna/GANs/blob/main/plots/train_ganY.svg)

Importantly, the training dataset determines the kind of examples the Generator will learn to emulate. If, for instance, our goal is to produce realistic-looking images of mice, we would supply our GAN with a dataset of mouse images.

The Generator’s goal is to produce examples that capture the data distribution of the training dataset. Object recognition models learn the patterns in images to discern an image’s content. The Generator can be thought of as the reverse of the process: rather than recognizing these patterns, it learns to synthesize them.

### Cost functions

*GANs* differ from conventional neural networks in two key respects. First, the cost function, $J$, of a traditional neural network is defined exclusively in terms of its own trainable parameters, $\theta$. Mathematically, this is expressed as $J(\theta)$. In contrast, *GANs* consist of two networks whose cost functions are dependent on *both* of the networks’ parameters. That is, the Generator’s cost function is $J^{(G)}(\theta^{(G)}, \theta^{(D)})$, and the Discriminator’s cost function is $J^{(D)}(\theta^{(G)}, \theta^{(D)})$. 

The second (related) difference is that a traditional neural network can tune all its parameters, $\theta$, during the training process. In a *GAN*, each network can tune only its own weights and biases. The Generator can tune only $\theta^{(G)}$, and the Discriminator can tune only $\theta^{(D)}$ during training. Accordingly, each network has control over only a part of what determines its loss.

### Training process

Because the Generator and Discriminator can tune only their own parameters and not each other’s, *GAN* training can be better described as a game, rather than optimization. *GAN* training ends when the two networks reach *Nash equilibrium*, a point in a game at which neither player can improve their situation by changing their strategy.

Mathematically, this occurs when the Generator cost $J^{(G)}(\theta^{(G)}, \theta^{(D)})$ is minimized with respect to the Generator’s trainable parameters $\theta(G)$ and, simultaneously, the Discriminator cost $J^{(D)}(\theta^{(G)}, \theta^{(D)})$ is minimized with respect to the parameters under this network’s control, $\theta(D)$.

### The Generator and the Discriminator

The Generator $(G)$ takes in a random noise vector $z$ and produces a fake example $x´$. Mathematically, $G(z) = x´$. The Discriminator $(D)$ is presented either with a real example $x$ or with a fake example $x´$; for each input, it outputs a value between $0$ and $1$ indicating the probability that the input is real. 

The Discriminator’s goal is to be as accurate as possible. For the real examples $x$, $D(x)$ seeks to be as close as possible to $1$ (label for the positive class). For fake examples $x´$, $D(x´)$ strives to be as close as possible to $0$ (label for the negative class). The Generator’s goal is the opposite. It seeks to fool the Discriminator by producing fake examples $x´$ that are indistinguishable from the real data in the training dataset. Mathematically, the Generator strives to produce fake examples $x´$ such that $D(x´)$ is as close to $1$ as possible.

### Confusion matrix

The Discriminator’s classifications can be expressed in terms of a confusion matrix, a tabular representation of all the possible outcomes in binary classification. In the case of the Discriminator, these are as follows:
- *True positive* - Real example correctly classified as real; $D(x) \approx 1$
- *False negative* - Real example incorrectly classified as fake; $D(x) \approx 0$
- *True negative* - Fake example correctly classified as fake; $D(x´) \approx 0$
- *False positive* - Fake example incorrectly classified as real; $D(x´) \approx 1$

### GAN training algorithm
***For*** each training iteration ***do***
1. Train the Discriminator
    - Take a random mini-batch of real examples: $x$.
    - Take a mini-batch of random noise vectors $z$ and generate a mini-batch of fake examples: $G(z) = x´$.
    - Compute the classification losses for $D(x)$ and $D(x´)$, and backpropagate the total error to update $\theta^{(D)}$ to minimize the classification loss.
2. Train the Generator
    - Take a mini-batch of random noise vectors z and generate a mini-batch of fake examples: $G(z) = x´$.
    - Compute the classification loss for $D(x´)$, and backpropagate the loss to update $\theta^{(G)}$ to maximize the classification loss.

***End for***

## **Deep Convolutional GANs - DCGANs**

Unlike the *GAN* architecture implemented previously, in *DCGANs*, both the Generator and Discriminator are implemented as convolutional neural networks - *CNNs*. In fact, One of the key techniques used in this case is *batch normalization*, which helps stabilize the training process by normalizing inputs at each layer where it is applied. 

### Batch normalization

*Normalization* is the scaling of data so that it has zero mean and unit variance. This is accomplished by taking each data point $x$, subtracting the mean $\mu$, and dividing the result by the standard deviation, $\sigma$, as shown:

$$
\hat{x} = \frac{x - \mu}{\sigma}
$$

*Normalization* has several advantages. Perhaps most important, it makes comparisons between features with vastly different scales easier and, by extension, makes the training process less sensitive to the scale of the features.

The insight behind *batch normalization* is that normalizing inputs alone may not go far enough when dealing with deep neural networks with many layers. As the input values flow through the network, from one layer to the next, they are scaled by the trainable parameters in each of those layers. And as the parameters get tuned by backpropagation, the distribution of each layer’s inputs is prone to change in subsequent training iterations, which destabilizes the learning process. In academia, this problem is known as *covariate shift*. Batch normalization solves it by scaling values in each minibatch by the mean and variance of that mini-batch.

The way batch normalization is computed differs in several respects from the simple normalization equation we presented earlier. Let $\mu_{B}$ be the mean of the mini-batch $B$, and $\sigma_{B}^{2}$ be the variance of the mini-batch $B$. The normalized value $\hat{x}$ is computed as:

$$
\hat{x} = \frac{x - \mu_{B}}{\sqrt{\sigma^{2} + \epsilon}}
$$

The term $\epsilon$ (epsilon) is added for numerical stability primarily to avoid division by zero. It is set to a small positive constant value, such as 0.001.

In batch normalization, we do not use these normalized values directly Instead, we multiply them by $\gamma$ (gamma) and add $\beta$ (beta) before passing them as inputs to the next layer; see equation bellow:

$$
y = \gamma \hat{x} + \beta
$$

Importantly, the terms $\beta$ and $\gamma$ are trainable parameters, which—just like weights and biases—are tuned during network training. The reason for this is that it may be beneficial for the intermediate input values to be standardized around a mean other than 0 and have a variance other than 1. Because $\beta$ and $\gamma$ are trainable, the network can learn what values work best.

Batch normalization limits the amount by which updating the parameters in the previous layers can affect the distribution of inputs received by the current layer. This decreases any unwanted interdependence between parameters across layers, which helps speed up the network training process and increase its robustness, especially when it comes to network parameter initialization.

## **Appendix** - The mathematical formulation of GANs


<!-- 
**Model Optimization**

Now that we have our loss functions, we can define the objective to optimize the parameters of both the generator and the discriminator.

*Training the Discriminator*

In practice, *GANs* are trained by alternating between optimizing the discriminator while keeping the generator fixed:

$$
V(G, D) = \mathbb{E}_{x \sim p_{\text{data}}}[\log(D(x))] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
$$

For an optimal discriminator $D^∗$:

$$
D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}
$$

*Training the Generator*

With $D^∗$ fixed, we focus on optimizing $G$. Using our substitution, the value function can be reformulated as:

$$
V(G, D^*) = -\log 4 + 2 \cdot D_{\text{JS}}(p_{\text{data}} \parallel p_g)
$$

Minimizing the Jensen-Shannon divergence, $D_{JS}$​, aligns the generator distribution $p_g$​ with the true data distribution $p_{data}$​. -->








## References
- [GANs in Action](https://www.google.de/books/edition/GANs_in_Action/HojvugEACAAJ?hl=en)