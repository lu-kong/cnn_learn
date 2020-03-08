WEEK 4 FACE RECOGNITION
===

live humain

# Face verifacation vs. face recognition
## Verification
- Input image, name/ID
- Output weather the input iamge is that of the claimed person

## Recongnition
- Has a database of K persons
- Get an input iamges
- Output ID if the image is any of the K persons (or "not recognized")
  
- **Note that Face Recognition is much more hard than Face Verification**



# One Shot Learning

Just one simple
Learning from one example to recognize the person again

## Learning a "similarity" function 

d(img1,img2) = degree of difference between images

$$
\begin{aligned}
    \text{If} \quad d(\text{img}1,\text{img}2) &\le \tau \quad \to \text{"same"}\\
    &> \tau \quad \to \text{"different"}
\end{aligned}
$$

*take input of a pair of iamges*

# Siamese network
First introduced by [Taigman et. al.,2014](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf)

Use the same parameters to calculate two iamges in order to compare with each other

How can we define a good loss funtion for this target?

# Triplet loss

Anchor image

First introduced by [Schroff et al.,2015](https://arxiv.org/abs/1503.03832)

Set a margin to make sure that we will not choose all zeros as parameters

## Loss function
A,P,N stand for three pictures which we define as a triplet of input of our loss function, which is:

- A : anchor
- P : positive *(that is, `high` similarity or the `same` person)*
- N : negative *(that is, `low` similarity and normally `different` person)*
$$
\begin{aligned}
   \mathcal{L}(A,P,N) &= \max\left( \left\|f(A)-f(P)\right\|^2 - \left\|f(A)-f(N)\right\|^2 + \alpha, 0\right)\\
   \mathcal{J} &= \sum_{i=1}^m\mathcal{L}\left(A^{(i)},P^{(i)},N^{(i)}\right)
\end{aligned}
$$

*ex. of training set*  
Training set is made up with $10K$ pictures of $1K$ different persons
- we need in fact multiple pictures of the same person so we can create the pair of $(A,P)$
- en average, each person have about 10 pictures

## How we choose the triplets A, P, N?
During training, if A, P, N are chosen `randomly`, $d(A,P) + \alpha \le d(A,N)$ is easily satisfied

We tend to choose those triplets that are `"hard"` to train on.
That is :
$$
\begin{aligned}
    d(A,P) + \alpha &\le d(A,N)\\
    \text{but, } \qquad d(A,P) + \alpha &\simeq d(A,N)
\end{aligned}
$$

In this way, our model will tend to make $d(A,P)$ larger and $d(A,N)$ smaller in converse.

## Face Verification and Binary Classification
Another way using binary classification to do the face varification that works also for our but.

Logistic regression unite to learning the similarity function
Cite: [Taigman et al., 2014](https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf))

Face verification supervised learning: Create a pair of pictures and predict wether these two pictures are belong to the same person.

# Neural Style Tranfer
- Content(C)
- Style(S)
- Generated image(G)
![](w4_001.png)

## Deep ConvNets Learning
> [Zeiler and Fergus.,2013, Visualizing and understanding convolutional networks](https://arxiv.org/abs/1311.2901)
1. Pick a unit *(that is, a neuron of this layer, or a filter of this layers)* in layer 1. Find the `nine` images patches that maximize the unit's activation
2. Repeat for other units
1. continue in deeper layers
- It comes to us that deeper the layer is, more complex will be the image it specifie.

## Cost function
> [Gatys et al.,2015 A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

We define a cost funtion like this:

$$
\mathcal{J}(G) = \alpha\mathcal{J}_{content}(C,G) \\+ \beta\mathcal{J}_{style}(S,G)
$$

### Find the generated image G
1. Initiate G randomly  
    $G: 100\times100\times3$
1. Use gradient descent to minimize $\mathcal{J}(G)$  
   $G\coloneqq G - \frac{\partial}{\partial G}\mathcal{J}(G)$

### Content cost function
![](w4_002.png)
The hidden layer chosen here is nether too shalow non too deep

### Style cost function
We don't care too much of of content of different images, in revenche, the style. We denote the style by detecting the high-light features of different channels of certain layers of activations for different images (S,G). Style is the combine of these high-light features of all channels. We consider both their absolute relation and relative relation.
#### Mraning
Say you are using layer l's activation to measure "style."  
Define style as `correlation` between activations across channels.  

The problem comes that how can we measure the degree of correlation?

#### Correlation
After finishing our [pre-step job](##Deep\ ConvNets\ Learning), we have already all the parameters for each layer of our basic `ConvNet`, we deteste the degree of correletion between two images by choosing on layer and calculate their activation in correspond.

Each layer of activations of the `style image` will give us slices in series in the sense of the style. That is, different slice of activation in layer $l$, gives us a serie of features that different channel corresponds.  

So when we look at the activations of the same layer but of our `generated image`, we will find also a series of features of different channels. If the two series of features can well correspond, we say that these two image are correleted and uncorreleted if not.

EX.  
if we choose the first hidden layer, that is, the first Conv layer normally, we will have a very clear feature of every channel because we know already how it looks like each filter.

While in deeper layer, different channels come from some series of acitivation that is usually much more complex. Two different style picture, even though of the same subject, will result in perhaps totally different features after being passed to the model. Because they donnot have the same style for the activations.

#### Style matrix
Let $a^{[l]}_{i,j,k}$
$$
\begin{aligned}
    \text{Let }a^{[l]}_{i,j,k} = \text{activation at }(i,j,k.)\\
G^{[l]} \text{ of }n^{[l]}_c \times n^{[l]}_c \text{ is called a style metrix of layer l defined by:}  \\
G^{[l](S)}_{kk^{\prime}} &= \sum_{i=1}^{n_H^{[l]}}\sum_{j=1}^{n_W^{[l]}}a_{ijk}^{[l](S)}a_{ijk^{\prime}}^{[l](S)}\\
G^{[l](G)}_{kk^{\prime}} &= \sum_{i=1}^{n_H^{[l]}}\sum_{j=1}^{n_W^{[l]}}a_{ijk}^{[l](G)}a_{ijk^{\prime}}^{[l](G)}\\
\mathcal{J}_{\text{style}}^{[l]}(S,G) &= \frac{1}{\left(2n_H^{[l]}n_W^{[l]}n_C^{[l]}\right)^2}\left\|G^{[l](S)}-G^{[l](G)}\right\|_F^2\\
&= \frac{1}{\left(2n_H^{[l]}n_W^{[l]}n_C^{[l]}\right)^2}\sum_{k}\sum_{k^{\prime}}\left(G^{[l](S)}_{kk^{\prime}}-G^{[l](G)}_{kk^{\prime}}\right)^2\\
\\

\mathcal{J}_{\text{style}}(S,G) &= \sum_{l}\lambda^{[l]}\mathcal{J}_{\text{style}}^{[l]}(S,G)\\
\\
\mathcal{J}(G) = \alpha\mathcal{J}&_{content}(C,G) \\&+ \beta\mathcal{J}_{style}(S,G)
\end{aligned} 
$$
