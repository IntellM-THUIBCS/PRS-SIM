---
layout: page
title: About
---
<!--
### [GitHub](https://github.com/cabooster/DeepCAD-RT) | [Paper](https://www.nature.com/articles/s41587-022-01450-8)
-->

## Content

- [Introduction](#introduction)
- [Results](#results)
- [Citation](#citation)

## Introduction

### Background
&emsp;Structured illumination microscopy is wide-used super-resolution technique for fluorescence imaging. 
Its based principle is to collect a series of wide-field images of the sample under different illumination patterns, e.g. the Morre fringe.
Since Morre fringe contains high spatial frequency information, by computational reconstruction, the high-order information can be retrieved, yield the improved resolution.
Typical for the linear fluorescence response, the resolution can be improved by 2 times.
Structured illumination microscopy is suitable for live-cell imaging due to its low photo-damage and high imaging speed compared to other super-resolution techniques, e.g. PALM/STORM and STED.
However, **the noise-induced artifact substantially degrades the quality of SIM images**. As shown in Fig. 1, although the detection noise is not obvious in WF image, 
since the SIM algorithm consist of the separation and re-combination of the low- and and high- order frequency information, the noise will be brought to high-frequency region and causing the ringing artifact.
This artifact will overwhelm the effective sample information in low-SNR situation, and strictly limits SIM application in long-term observation of the cellular organelles and dynamics.


<center><img src="../images/Demo-website-artifact.png?raw=true" width="1000" align="middle" /></center>
<center>Figure 1 | Detection noise significantly degrades the SIM quality.</center>


### Methods

&emsp;In this work, we proposed RES-SIM, a image recorruption based self-supervised denoising technique for SIM. 
We used to **The mix Poisson-Gaussian model.** to model the detection noise, which is universal in microscopy images. 
The key of RES-SIM is to use image recorruption strategy <sup>[1]</sup> to create a super-resolution image pair statisifying the "independent" criteria. 
Therefore, based on N2N<sup>[2]</sup> mechanism, this image pair can be set as the input and target to train a denoising network.
By iteratively optimized loss function, the model can finally gain the denoising capability.<br>

&emsp;The diagram of RES-SIM is shown as Fig. 2. To create the training dataset. We first employed the image recorruption strategy to each raw image in the SIM stacks and generation two paired low-resolution image stacks.
Then we applied the conventional SIM reconstruction algorithm to both stack and acquire two paired super-resolution images. And finally we categortized these two images as the training and the target of the network. 
By repeating these steps to each raw image stacks, we can acquire a series of corrupted image pairs, and formulate the final training dataset. 
To enrich the dataset, we can recorrupt the same images with different parameters to generate more images. Typical, 30~50 raw SIM stacks is enough for a successful denoising network.<br>

&emsp; During the inference phase, we first applied conventional SIM algorithm on the noisy raw SIM image stack to acquire a super-resolution noisy SIM iamge, then input it to the pre-trained network to output the 
denoised super-resolution image.<br>

&emsp; RES-SIM is a self-supervised denoising methods since only the low-SNR images are needed to create the training dataset and pre-trained model is then applied to denoise themselves. 
Compared to other deep-learning based denoising method such as N2N, no repeated acquisition is required, making RES-SIM suitable for fast-moving samples.
For time-lapsing videos with enough frames, RES-SIM is capable to train the model and perform the denoising with only the collected data itself, enabling the potential discovery of the novel structure.

<center><img src="../images/principle.png?raw=true" width="1000" align="middle" /></center>
<center>Figure 2 | Diagram of RES-SIM</center>

### Our Contribution
&emsp; We present self-supervised method **RES-SIM** to denoise SIM images, which utilzes only low-SNR data themselves to train the network and provides an artifact-free, SNR-improved, super-resolution image.
RES-SIM is able to achieve comparable image quality as GT images with extremely low collected photons, which reduces the photo-toxicity (low excitation power) and improve imaging speed (short exposure time). 
Benefiting from this, RES-SIM enables long term observation of the cells sensitive to photo-damage, e.g. the growing SUM-159 cells. Since organelles in RES-SIM ehanced image resolution is resolved at high-defination, 
the segmentation and tracking algorithm is appliable to further investigate their biological insight. Based on RES-SIM result, we discovered a novel CCPs population phenomenon and the interactions between the CCPs and cytoskeleton.
RES-SIM is also operated in offline mode for time-lapsed data, which utilized only the noisy data to train the network and performed the denoise themselves. Benefiting from this, no pre-trained model is needed, 
which is potential to discover the unseend biological structures.

## Results
Benefitting from artifact removal and the contrast enhancement by RES-SIM, more detailed information of the biological structure can be resolved. Here we demonstrated some representive RES-SIM results.<br>

<center><h3>1. RES-SIM massively improves the image quality on multiple organelles. <br> (scalebar: 2&mu;m )</h3></center>
<center><img src="../images/Figure-website-about1.png?raw=true" width="800" align="middle"></center>


<center><h3>2. RES-SIM enables long-term volumetric imaging with ultra low excitation power. <br> (scalebar: 5&mu;m)</h3></center>
<center><img src="../images/Figure-website-about2.png?raw=true" width="800" align="middle"></center>

<center><h3>3. RES-SIM enables precise segmentation for studying the biological insight. <br>(scalebar: 5&mu;m, 1&mu;m )</h3></center>
<center><img src="../images/Figure-website-about3.png?raw=true" width="1000" align="middle"></center>
 
<!--
## Results
RES-SIM is compatible with Mul

<center><h3>1. DeepCAD-RT massively improves the imaging SNR of neuronal population recordings in the zebrafish brain</h3></center>

<center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/gallery_zebra.png?raw=true" width="850" align="middle"></center>

<center><h3>2. DeepCAD-RT reveals the 3D migration of neutrophils in vivo after acute brain injury</h3></center>

<center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/gallery_NP.png?raw=true" width="850" align="middle"></center>

<center><h3>3. DeepCAD-RT reveals the ATP (Adenosine 5’-triphosphate) dynamics of astrocytes in 3D after laser-induced brain injury</h3></center>

<center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/gallery_ATP.png?raw=true" width="850" align="middle"></center>


<center>More demo images and videos are demonstrated in <a href='https://cabooster.github.io/DeepCAD-RT/Gallery/'>Gallery</a>. More details please refer to <a href='https://www.nature.com/articles/s41587-022-01450-8'>the companion paper</a></center>.
-->


## Citation
[1] Pang, Tongyao, et al. "Recorrupted-to-recorrupted: unsupervised deep learning for image denoising." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021. <br>
[2] Lehtinen J, Munkberg J, Hasselgren J, et al. Noise2Noise: Learning image restoration without clean data[J]. arXiv preprint arXiv:1803.04189, 2018. <br>