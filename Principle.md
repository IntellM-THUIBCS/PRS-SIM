---
layout: page
title: Principle
---
<!--
### [GitHub](https://github.com/cabooster/DeepCAD-RT) | [Paper](https://www.nature.com/articles/s41587-022-01450-8)
-->

<h2 style="color:white;" id="Method">Method</h2>

The key of PRS-SIM is applying a pixel-realignment strategy to create the training dataset for the super-resolution SIM denoising network.
The raw training dataset of PRS-SIM consists of a series of low-SNR raw SIM image groups. 
Each individual image in a group is a WF image under a specific illumination pattern
 (e.g. 3-orientation × 3-phase for 2D/TIRF-SIM and 3-orientation × 5-phase × Z-slice for 3D-SIM). 
 For each raw SIM image group, the generation of the training dataset of PRS-SIM models mainly takes the following steps:

+ Each raw image is divided into 4 sub-images by applying a 2×2 down-sampler and formed four sub-image groups.
+ The augmented four sub-image groups are re-up sampled into the original size with the nearest interpolation.
+ Based on the position of the valid pixel in each 2×2 cell, a sub-pixel translation is applied to each raw image, which guarantees that they are well spatially calibrated with each other.
+ The generated sub-images groups are reconstructed into four noisy SIM images by applying the conventional SIM algorithm. 
+ Then several image patched pairs are augmented by randomly selecting two out of four noisy SIM images as the input and target, and applying ordinary data augmentation operations, e.g., random cropping, flipping and rotation.

Then in the inference phase, we firstly apply conventional SIM algorithm on the noise raw SIM images to generate a SR image, 
then input it to the pre-trained model. The final denoised SR image will be output by the network.


<center><img src="../images/Figure-website-demo.png?raw=true" width="1050" align="center" /></center>

<h2 style="color:white;" id="Theory">Theory</h2>

In this part, we provide a brief proof of  effectiveness of PRS-SIM for super-resolution denoising SIM images
(more details will be discussed in our manuscript).

<h3 style="color:white;" >Denoising regular images without clean data</h3>

To prove PRS-SIM, we firstly provided a brief review of the classical deep-learning based image denoising method N2N<sup>[1]</sup>,
which utilized only two independently captured noisy images of the same sample to train the network.
Based on the simplified noise model in previous section, the detected noisy image $\mathbf{y}$ corresponding to the clean object $\mathbf{x}$ can be modelled as 

<center> 
$$
\mathbf{y} = \mathbf{x} + \mathbf{n} \tag{3}
$$
</center>
where, $\mathbf{n}$ denotes the random noise following normal distribution $N(0,\mathcal{\sigma^2})$. 
The goal of image denoising task is to retrieve clean image $\mathbf{x}$ from noisy input $\mathbf{y}$.

For supervised denoising neural networks, 
the training dataset consists of a series of noisy/clean image pairs, 
and the objective function is defined as

<center> 
$$
min || \phi(\mathbf{y})-\mathbf{x}|| \tag{4}
$$
</center>
where $\phi (\cdot)$ denotes the neural network.

Let Let $\mathbf{y_1}=\mathbf{x}+\mathbf{n_1}$ and $\mathbf{y_2}=\mathbf{x}+\mathbf{n_2}$ two noisy image corresponding to the same
ground-truth $\mathbf{x}$. 
Due to the property of independent acquisition, $\mathbf{n_1}$ and $\mathbf{n_2}$ follows independent normal distribution.
Then the mathematical expectation of objective function of N2N is formulated as:

<center> 
$$
\begin{align}
    E\{||\phi(\mathbf{y_1})-\mathbf{y_2}||^2_2\} &= E\{||\phi(\mathbf{y_1})-\mathbf{x}+\mathbf{n_2}||^2_2\}\\
    &= E\{||\phi(\mathbf{y_1})-\mathbf{x}||^2_2-2E\{\phi(\mathbf{y_1})-\mathbf{x}\cdot\mathbf{n_2}\}
    +E\{\mathbf{n_2^2}\}\\
    &=  E\{||\phi(\mathbf{y_1})-\mathbf{x}||^2_2-2E\{\phi(\mathbf{y_1})-\mathbf{x}\}\cdot E\{\mathbf{n_2}\}
    +E\{\mathbf{n_2^2}\}\\
    & =  E\{||\phi(\mathbf{y_1})-\mathbf{x}||^2_2+\sigma^2
\end{align}\tag{5}
$$
</center> 

In Eq. 5, $E(||\phi(\mathbf{y_1})-\mathbf{x}||^2_2)$ is identical with the objective function of supervised scheme to trained with noise-free ground-truth,
and $\sigma^2$ is a constant. 
Therefore, N2N training scheme is able to gain comparable denoising performance as supervised learning.  
Because of its ingenious theory and superior denoising performance, N2N has become a millstone algorithm and enlightened several subsequent denoising schemes.

<h3 style="color:white;" >Self-supervised denoising with similar scenario</h3>

Although N2N has shown great denoising performance on either natural images or microscopic images, 
the requirement of duplicated captures of the same specimen strictly limits its application. 
To further develop denoising techniques applicable for single-captured SIM images, 
we investigated data augment strategy termed neighbor2neighbor<sup>[2]</sup>, 
which exploiting the similarity between adjacent pixels to create the training dataset. 
Specifically, let $\mathbf{y_A}$ and $\mathbf{y_B}$ denote two sub-images extracted from the same noisy image $\mathbf{y}$ corresponding to the ground-truth $\mathbf{x}$, 
it is easy to prove that the noise in $\mathbf{y_A}$ and $\mathbf{y_B}$ follows the independent zero-mean Gaussian distribution. 
So their conditional mathematical expectation is represented as:
<center> 
$$
\begin{align}
E_{\mathbf{y_A}|\mathbf{x}}(\mathbf{y_A}) & =\mathbf{x}\\
E_{\mathbf{y_B}|\mathbf{x}}(\mathbf{y_B}) & =\mathbf{x}+\varepsilon
\end{align}\tag{6}
$$
</center> 
where $\varepsilon$ denotes the margin between the underlying ground-truth of $\mathbf{y_A}$ and $\mathbf{y_B}$. 

Therefore, the mathematical expectation of the objective function to training the neural network with $\mathbf{y_A}$ (input) and $\mathbf{y_B} (target) can be represented as:
<center> 
$$
\begin{align}
E_{\mathbf{y_A},\mathbf{y_B}|\mathbf{x}}\{||\phi(\mathbf{y_A})-\mathbf{y_B}||^2_2\} &= E_{\mathbf{y_A}|\mathbf{x}}\{||\phi(\mathbf{y_A})-\mathbf{x}+\mathbf{x}-\mathbf{y_B}||^2_2\}\\
&= E_{\mathbf{y_A}|\mathbf{x}}\{||\phi(\mathbf{y_A})-\mathbf{x}||^2_2\} +\sigma^2 -2 \varepsilon E_{\mathbf{y_A}|\mathbf{x}}(\phi(\mathbf{y_A})-\mathbf{x})
\end{align}\tag{7}
$$
</center> 

Although $\varepsilon$ is a non-zero variable, since the sub-images $\mathbf{y_A}$ and $\mathbf{y_B}$ is extracted by the adjacent pixels, 
the exact value of $\varepsilon$ should be comparable small, 
so that the item $2 \varepsilon E_{\mathbf{y_A}|\mathbf{x}}(\phi(\mathbf{y_A})-\mathbf{x})$ is close to zero. 
Since $E_{\mathbf{y_A},\mathbf{y_B},\mathbf{x}}(\cdot)=E_{\mathbf{y_A},\mathbf{y_B}|\mathbf{x}}(\cdot)E_{\mathbf{x}}(\cdot)$, we further have:

<center> 
$$
E_{\mathbf{y_A},\mathbf{y_B}}\{||\phi(\mathbf{y_A})-\mathbf{y_B}||^2_2\} \approx E_{\mathbf{y_A},\mathbf{y_B},\mathbf{x}}\{||\phi(\mathbf{y_A})-\mathbf{x}||^2_2\}+\sigma^2\tag{8}
$$
</center> 

Concluded from Eq.8, to train the neural network with $\mathbf{y_A}$ and $\mathbf{y_B}$ is an acceptable approximation to the supervised training with clean data.

<h3 style="color:white;" >Pixel-realignment strategy for SIM denoising</h3>

After introducing the self-supervised denoising strategy with similar scenario,
we make a further step in this section to incorporate it with SIM reconstruction and denoising.
The basic theory of SIM imaging can be referred from [3,4]. From a series raw images $\mathbf{y_1}$, $\mathbf{y_2}$, $\dots$, $\mathbf{y_n}$,
the super-resolution recorrupted image $S(\mathbf{r})$ can be represented as:<br>
<center>
$$
S(\mathbf{r})=\mathcal{F}^{-1}(\frac{\sum_{p,m}\tilde{D}_{p,m}(\mathbf{k_r}-m\mathbf{p})\cdot OTF(\mathbf{k_r})}{\sum_{p,m}||OTF(\mathbf{k_r}-m\mathbf{p})||^2+\omega^2})
\tag{9}
$$
</center>

All the computation operations for $S(\mathbf{r})$, including matrix multiplication, (inversed) Fourier transformation, Wiener filtering, 
are all linear operations with no offset. 
Let $\mathbf{Y_A}$ and $\mathbf{Y_B}$ denotes the super-resolution images reconstructed from raw SIM image group $\{\mathbf{y_{A1}},\mathbf{y_{A2}},\dots,\mathbf{y_{An}}\}$, 
and $\{\mathbf{y_{B1}},\mathbf{y_{B2}},\dots,\mathbf{y_{Bn}}\}$, respectively. We have:
<center>
$$
\begin{align}
\mathbf{Y_A} &= c_1\mathit{TU}(\mathbf{y_{A1}})+c_2\mathit{TU}(\mathbf{y_{A2}})+\dots+c_n\mathit{TU}(\mathbf{y_{An}})\\
\mathbf{Y_B} &= c_1\mathit{TU}(\mathbf{y_{B1}})+c_2\mathit{TU}(\mathbf{y_{B2}})+\dots+c_n\mathit{TU}(\mathbf{y_{Bn}})
\end{align}\tag{10}
$$
</center>
where $c_i$ denotes the coefficient for conventional SIM reconstruction and $\mathit{TU}(\cdot)$ denotes the integrated operator of translation and up-sampling.

From Eq. 10, it is easy to proof that the noise distribution of the reconstructed image $\mathbf{Y_A}$ and $\mathbf{Y_B}$ also follows independent 0-mean normal distribution.
Since the raw image $\mathbf{y_{Ai}}$ and $\mathbf{y_{Bi}}$ are sub-sampled from the same raw image, they have enough similarity. 
The the effectiveness of training the super-resolution denoising neural network with $\mathbf{y_{Ai}}$ and $\mathbf{y_{Bi}}$ is proved.
 

<h2 style="color:white;" id="Reference">Reference</h2>
<p>
[1] Lehtinen J, Munkberg J, Hasselgren J, et al. Noise2Noise: Learning image restoration without clean data. arXiv preprint arXiv:1803.04189, 2018. <br>
[2] Huang, T., Li, S., Jia, X., Lu, H., & Liu, J. (2022). Neighbor2neighbor: A self-supervised framework for deep image denoising. IEEE Transactions on Image Processing, 31, 4023-4038.<br>
[3] Gustafsson, M. G. (2000). Surpassing the lateral resolution limit by a factor of two using structured illumination microscopy. Journal of microscopy, 198(2), 82-87.<br>
[4] Gustafsson, M. G., Shao, L., Carlton, P. M., Wang, C. R., Golubovskaya, I. N., Cande, W. Z., ... & Sedat, J. W. (2008). Three-dimensional resolution doubling in wide-field fluorescence microscopy by structured illumination. Biophysical journal, 94(12), 4957-4970.<br>
</p>

