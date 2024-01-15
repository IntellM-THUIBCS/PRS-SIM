# Introduction of the Matlab code for pixel-realignment

***

To apply pixel-realignment strategy to the raw SIM images, just execute the script 'Create_training_dataset.m' in Matlab.

The raw data should be saved in folder 'raw_data' and the realigned images will be output in a new folder 'realigned_data'.

The raw image file is saved in '.mrc' format, which is a common-used format for saving microscopy data. The example IO code for reading '.mrc' file is located in folder 'XxMatlabUtils'.
The detailed information about the image, e.g., the pixel number, the pixel size, the time point number is saved in the variable 'header'.

To further generate SR-SIM image, please apply conventional SIM algorithm on the realigned images, which can be acquired from several the open-source packages, e.g. 3-beam-SIM<sup>[1]</sup> (https://github.com/scopetools/cudasirecon), fairSIM<sup>[2]</sup> (https://github.com/fairSIM/fairSIM), Open-3DSIM<sup>[3]</sup> (https://github.com/Cao-ruijie/Open3DSIM), and PCA-SIM<sup>[4]</sup> (https://link.springer.com/article/10.1186/s43593-022-00035-x).

Due to the storage limitation of Github, the example raw SIM data of Microtubules (TIRF-SIM) and Lysosome (3D-SIM) are uploaded on Google drive https://drive.google.com/drive/folders/1SW7Lt3G5I-D6D-7KFyruIoZX7Cc9Nm_V?usp=sharing.

The detailed tutorial of PRS-SIM is available on our website https://intellm-thuibcs.github.io/PRS-SIM.


Citation:

[1] Gustafsson, Mats GL, et al. "Three-dimensional resolution doubling in wide-field fluorescence microscopy by structured illumination." Biophysical journal 94.12 (2008): 4957-4970.<br>

[2] Müller, Marcel, et al. "Open-source image reconstruction of super-resolution structured illumination microscopy data in ImageJ." Nature communications 7.1 (2016): 10980.<br>

[3] Cao, Ruijie, et al. "Open-3DSIM: an open-source three-dimensional structured illumination microscopy reconstruction platform." Nature Methods 20.8 (2023): 1183-1186.<br>

[4] Qian, Jiaming, et al. "Structured illumination microscopy based on principal component analysis." eLight 3.1 (2023): 4.3.<br>
