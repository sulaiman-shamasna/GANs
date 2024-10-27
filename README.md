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


1. **Training the Discriminator**.
    - Take a random real example *X* from the training dataset.
    - Get a new random noise vector *Z* and, using the Generator network, synthesize a fake example *X´*.
    - Use the Discriminator network to classify *X* and *X´*.
    - Compute the classification errors and backpropagate the total error to update the Discriminator weights and biases, seeking to *minimize* the classification errors.
2. **Training the Generator**.
    - Get a new random noise vector *Z* and, using the Generator network, synthesize a fake example *X´*.
    - Use the Discriminator network to classify *X´*.
    - Compute the classification error and backpropagate the error to update the Generator weights and biases, seeking to *maximize* the Discriminator's error.

## Reaching Equilibrium
In a GAN, the two networks have competing objectives: when one network gets better, th other gets worse. How do we determine when to stop?

From Game Theory point of view, this step may be recognized as a *zero-sum game*—a situation in which one player’s gains equal the other player’s losses. When one player improves by a certain amount, the other player worsens by the same amount. All *zero-sum games* have a *Nash equilibrium*, a point at which neither player can improve their situation or payoff by changing their actions.

GAN reaches Nash equilibrium when the following conditions are met:
- The Generator produces fake examples that are indistinguishable from the real data in the training dataset.
- The Discriminator can at best randomly guess whether a particular example is real or fake (that is, make a 50/50 guess whether an example is real).

## References
- [GANs in Action](https://www.google.de/books/edition/GANs_in_Action/HojvugEACAAJ?hl=en)