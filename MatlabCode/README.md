To apply pixel-realignment strategy to the raw SIM images, just execute the script 'Create_training_dataset.m' in Matlab.

The raw data should be saved in folder 'raw_data' and the realigned images will be output in a new folder 'realigned_data'.

The raw image file is saved in '.mrc' format, which is a common-used format for saving microscopy data. The example IO code for reading '.mrc' file is located in folder 'XxMatlabUtils'.
The detailed information about the image, e.g., the pixel number, the pixel size, the time point number is saved in the variable 'header'.

To further generate SR-SIM image, please apply conventional SIM algorithm on the realigned images, which can be acquired from several the open-source packages, e.g. fairSIM<sup>[1]</sup>, Hifi-SIM<sup>[2]</sup>, and OpenSIM<sup>[3]</sup>.

Due to the storage limitation of Github, the example raw SIM data of Microtubules (TIRF-SIM) and Lysosome (3D-SIM) are uploaded on Google drive https://drive.google.com/drive/folders/1SW7Lt3G5I-D6D-7KFyruIoZX7Cc9Nm_V?usp=sharing .

The detailed tutorial of PRS-SIM is available on our website .


Citation:
[1] Müller M, Mönkemöller V, Hennig S, et al. Open-source image reconstruction of super-resolution structured illumination microscopy data in ImageJ[J]. Nature communications, 2016, 7(1): 1-6.<br>
[2] Wen G, Li S, Wang L, et al. High-fidelity structured illumination microscopy by point-spread-function engineering[J]. Light: Science & Applications, 2021, 10(1): 1-12.<br>
[3] Lal A, Shan C, Xi P. Structured illumination microscopy image reconstruction algorithm[J]. IEEE Journal of Selected Topics in Quantum Electronics, 2016, 22(4): 50-63.<br>
