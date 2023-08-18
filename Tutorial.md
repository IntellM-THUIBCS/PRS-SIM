---
layout: page
title: Tutorial
---

<h2 style="color:white;">Content</h2>

<ul>
  <li><a href="#Matlab">Matlab code for generatint training dataset</a></li>
  <li><a href="#Python">Python code for network training and inference</a></li>
  <li><a href="#Fiji">Fiji plugin</a></li>
</ul>

<h2 style="color:white;" id="Matlab">1. Matlab source code</h2>
<h3 style="color:white;">Create the training dataset</h3>
<p>
Each data item in PRS-SIM training dataset is a super-resolution (SR) image pair generated from a low-SNR raw image group (3*3 for a 2D/TIRF-SIM image, 3*5*z_num for a 3D-SIM volume, and 1*3 for a LLS-SIM slice). 
For each raw image group, we first employ the pixel-realignment strategy form 4 realigned image stack, then apply convention SIM algorithm to generate 4 SR images.
During the training process, 2 of which are randomly selected and arranged as the input data and target data, respectively.
By repeating this operations to each image stack, the final training dataset is formulated. 
Typically, ~30 individual image stacks is adequate for a successful training.
</p>

<p>
To create the training dataset, please execute the file "Create_training_dataset.m" in the Matlab command window as
</p>

<code style="background-color:#393939;">
run('Create_training_dataset.m');
</code><br>
<p>
Several parameters need to assign in the file "Create_training_dataset.m", including the parameter for file IO:
</p>
<table border="1">
	<tr>
		<td><font color=black>Parameter name</font></td>
		<td><font color=black>Description</font></td>
	</tr>
	<tr>
		<td><font color=white>Smpl_name</font></td>
		<td><font color=white>The name of the sample.</font></td>
	</tr>
	<tr>
		<td><font color=black>File_dir</font></td>
		<td><font color=black>The directory to save the raw data, default is the 'data' folder in current directory. Each individual image file should be '.mrc' format. </font></td>
	</tr>
	<tr>
		<td><font color=white>Save_file_dir</font></td>
		<td><font color=white>The directory to save the data, default is the 'data' folder in current directory.</font></td>
	</tr>
</table>

<p>
After running the script, the 4 realigned images will be saved in the same sub-folder (also '.mrc' format).
To further reconstruct the super-resolution SIM images, conventional SIM reconstruction algorithm can also be employed, which can be acquired 
from several the open-source packages, e.g. fairSIM<sup>[1]</sup>, Hifi-SIM<sup>[2]</sup>, and OpenSIM<sup>[3]</sup>.
</p>

<h2 style="color:white;" id="Python">2. Python source code</h2>

<p>
The python code is written for training and inference of the network based on the Pytorch framework. To accelerate the training process, a GPU is required.
</p>

<h3 style="color:white;">Environment installation</h3>

<p>
To avoid the version incompatiblility, we highly recommend to install PRS-SIM in a CONDA vitual environment as 
</p>

<code style="background-color:#393939;">
conda create -n PRSSIM python=3.7
</code><br>
<p>

To install Pytorch framework, please follow the instruction on the Pytorch website <a href="https://pytorch.org/get-started/locally/#windows-anaconda">https://pytorch.org/get-started/locally/#windows-anaconda</a>
based on the exact type and version the hardware of your computer. 
An example command for Pytorch is (may be replaced by your own configuration)

</p>
<code style="background-color:#393939;">
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=11.3 -c pytorch -c conda-forge
</code>

Other required packages are listed in file 'requirements.txt', which can be installed by pip command as.

<code style="background-color:#393939;">
pip install -r requirements.txt
</code>
 
Pleased note that the suggested version information is only validated on our working station, and some modification is possibly needed for your specific environment. 

<p>
To begin the training or inference processing, just activate the vitual environment  as:
</p>
<code style="background-color:#393939;">
conda activate PRSSIM<br>
</code><br>

<h3 style="color:white;">Network training</h3>
<p>
The default folder to save SIM data is './SIM_data/smpl_name'.Before the training, we need to organize the sub-folder in the following architecture.

</p>
* Cell X
   * view1.tif
   * view2.tif
   * view3.tif
   * view4.tif
<p>
where 'viewX.tif' denotes the reailgned SIM images, which is reconstructed from the reailgned raw images by SIM algorithm, e.g. fairSIM<sup>[1]</sup>, Hifi-SIM<sup>[2]</sup>, and Open-(3D)SIM<sup>[3][4]</sup>.<br>
</p>

<p>
Our code is developed based on open source framework KAIR (<a href="https://github.com/cszn/KAIR">https://github.com/cszn/KAIR</a>), in which the interface for multiple network, model and dataset is provided. 
The training condition is defined in the '.json' file located in folder 'options'. To specific your own training, just modify the parameters in the corresponding file. 
Main tubable parameters for training are listed in the following table, and other customized parameters can be added. <br>
</p>

<table border="1">
	<tr>
		<td><font color=black>Parameter name</font></td>
		<td><font color=black>Default</font></td>
		<td><font color=black>Description</font></td>
	</tr>
	<tr>
		<td><font color=white>Optimizer name</font></td>
		<td><font color=white>Adam</font></td>
		<td><font color=white>The type of the optimizer</font></td>
	</tr>
	<tr>
		<td><font color=black>Initial learning rate</font></td>
		<td><font color=black>1e-4</font></td>
		<td><font color=black>The initial learning rate</font></td>
	</tr>
	<tr>
		<td><font color=white>Scheduler name</font></td>
		<td><font color=white>Multistep</font></td>
		<td><font color=white>The type of the scheduler</font></td>
	</tr>
	<tr>
		<td><font color=black>Milestones</font></td>
		<td><font color=black>[30000,40000,50000,60000]</font></td>
		<td><font color=black>The iteration point to decrease the learning rate</font></td>
	</tr>
	<tr>
		<td><font color=white>Scheduler gamma</font></td>
		<td><font color=white>0.5</font></td>
		<td><font color=white>The decay ratio of initial learning at each scheduler point</font></td>
	</tr>
	<tr>
		<td><font color=black>Checkpoints</font></td>
		<td><font color=black>10000</font></td>
		<td><font color=black>The iteration point to perform the validation</font></td>
	</tr>
</table>


<p>
Although we only used U-net as the network backbone in this work, it is usable to employ other network archetecture, such as RCAN and RDN. We also provided the interfaces for these network. 
If your want to employ other network, 
just create a new '.py' file for network and a '.json' file for training condition, and then add a link by adding an item in 'select_model.py'. <br>
</p>
<p>
To run the training script, the following parameters need to be assigned as:<br>
</p>

<table>
	<tr>
		<td><font color=black>Parameter name</font></td>
		<td><font color=black>Type</font></td>
		<td><font color=black>Description</font></td>
	</tr>
	<tr>
		<td><font color=white>smpl_dir</font></td>
		<td><font color=white>str</font></td>
		<td><font color=white> The local directory saving the dataset, default is the folder './dataset' in current directory</font></td>
	</tr>
	<tr>
		<td><font color=black>smpl_name</font></td>
		<td><font color=black>str</font></td>
		<td><font color=black>The name of the sub folder saving the sample data</font></td>
	</tr>
	<tr>
		<td><font color=white>gpu_id</font></td>
		<td><font color=white>str</font></td>
		<td><font color=white>  The specific gpu device assigned for training, default is '0'</font></td>
	</tr>
	<tr>
		<td><font color=black>network_type</font></td>
		<td><font color=black>str</font></td>
		<td><font color=black>The type of network used in the model, e.g. 'unet', 'rcan', 'rdn', default is 'unet'</font></td>
	</tr>
	<tr>
		<td><font color=white>gpu_id</font></td>
		<td><font color=white>str</font></td>
		<td><font color=white>  The specific gpu device assigned for training, default is '0'</font></td>
	</tr>
	<tr>
		<td><font color=black>save_suffix</font></td>
		<td><font color=black>str</font></td>
		<td><font color=black>The suffix of the saving model</font></td>
	</tr>
	<tr>
		<td><font color=white>preload_data_flag</font></td>
		<td><font color=white>bool</font></td>
		<td><font color=white>The logial variable indicating whether to pre-load all data in memory to accerlate the training, default is False</font></td>
	</tr>
	<tr>
		<td><font color=black>max_iter</font></td>
		<td><font color=black>int</font></td>
		<td><font color=black>The maximum iteration to train the model </font></td>
	</tr>
</table>

<p>
The example code for training is :
</p>

<code style="background-color:#393939;">
cd ./<br>
python Main_train.py --gpu_id 0 --smpl_dir ./SIM_data --smpl_name Microtubules --net_type unet --save_suffix _0 --test_patch_size 128 --max_iter 100000 --preload_data_flag<br>
</code><br>

<h3 style="color:white;">Denoising</h3>

<p>
To implement the denoising with the trained model, the following parameters need to be assigned:
</p>


<table>
	<tr>
		<td><font color=black>Parameter name</font></td>
		<td><font color=black>Type</font></td>
		<td><font color=black>Description</font></td>
	</tr>
	<tr>
		<td><font color=white>smpl_dir</font></td>
		<td><font color=white>str</font></td>
		<td><font color=white> The local directory saving the dataset, default is the folder './dataset' in current directory</font></td>
	</tr>
	<tr>
		<td><font color=black>smpl_name</font></td>
		<td><font color=black>str</font></td>
		<td><font color=black>The name of the sub folder saving the sample data</font></td>
	</tr>
	<tr>
		<td><font color=white>data_format</font></td>
		<td><font color=white>str</font></td>
		<td><font color=white>  The format of the raw data file, e.g. 'npy', 'mat', 'tif', default is 'npy'</font></td>
	</tr>
	<tr>
		<td><font color=black>network_type</font></td>
		<td><font color=black>str</font></td>
		<td><font color=black>The type of network used in the model, e.g. 'unet', 'rcan', 'rdn', default is 'unet'</font></td>
	</tr>
	<tr>
		<td><font color=white>gpu_id</font></td>
		<td><font color=white>str</font></td>
		<td><font color=white>  The specific gpu device assigned for training, default is '0'</font></td>
	</tr>
	<tr>
		<td><font color=black>model_name</font></td>
		<td><font color=black>str</font></td>
		<td><font color=black>The name of the model files (in '.pth' format)</font></td>
	</tr>
	<tr>
		<td><font color=white>overlap_ratio</font></td>
		<td><font color=white>int</font></td>
		<td><font color=white>The ratio between adjacent patches </font></td>
	</tr>
	<tr>
		<td><font color=black>test_patch_size</font></td>
		<td><font color=black>int</font></td>
		<td><font color=black>The size of the cropped region of the test images, default is 1000 </font></td>
	</tr>
	<tr>
		<td><font color=white>model_patch_size</font></td>
		<td><font color=white>int</font></td>
		<td><font color=white>The size of the mini image patch input to the network, default is 128</font></td>
	</tr>
</table>
<p>
An example code to perform the denoising is:
</p>

<code style="background-color:#393939;">
python Main_test_3D.py --gpu_id 0 --smpl_dir ./SIM_data --smpl_name Lyso_test --net_type unet --model_name 100000_G --test_patch_size 1004 --model_patch_size 128 --overlap_ratio 0.2
</code><br>

<h3 style="color:white;">Examples</h3>
<p>
A representative training dataset and inference dataset are located in the 'dataset' of current repository. To provide a intuitive view of our software, 
we also uploaded the corresponding pre-trained model for this dataset. The example training and denoised images are shown in the following figure.
</p>

<img src="../images/Demo-website-training.png?raw=true" width="600" align="middle">

<h2 style="color:white;" id="Fiji">3. Fiji plugin</h2>
<p>
To make our work convenient for more biological researchers, we also provided a ready-to-use Fiji plugin for SIM denoising. The detailed instruction of our plugin is provided in the following section.<br>
(<i>More functions, e.g. the training 3D model in RES-SIM plugin, are under development, and will be released in the further update.</i>)
</p>

</p>

<h3 style="color:white;">Installation</h3>

1. Copy `./Fiji-plugin/jars/*` and `./Fiji-plugin/plugins/*` to your root path of Fiji `/*/Fiji.app/`.
2. Restart Fiji
3. Start the PRS-SIM Fiji Plugin:

    <div align=center><img src="../images/access.png" alt="Access to PRS-SIM Fiji Plugin" />

<h3 style="color:white;">Set CPU/GPU and TensorFlow version</h3>

The PRS-SIM Fiji plugin was developed based on TensorFlow-Java 1.15.0, which is compatible with CUDA version of 10.0 and cuDNN version of 7.5.1. 
Before running the plugin with GPU, you should select and install Tensorflow GPU in Fiji by following steps:

1. Open `.Edit > Options > Tensorflow*`, and choose the version matching your model or setting.
2. Wait until a message pops up telling you that the library was installed.
3. If choose TensorFlow GPU version, please install these NVIDIA requirements to run TensorFlow with GPU support as described in `.Edit > Options > Tensorflow*`, or GPU support will not work.
4. If run ImageJ on Windows machines, please copy `./Fiji-plugin/lib/win64/tensorflow_jni.dll` to `C:/Windows/System32/`.
5. Restart Fiji.

	<div align=center><img src="../images/tensorflow.png" alt="Edit > Option > Tensorflow" />

Note that GPU Support of TensorFlow in ImageJ is only available for Linux and Windows operating system with NVIDIA graphics cards at the present. 
Therefore, the PRS-SIM plugin currently runs only with CPU for macOS.


<h3 style="color:white;">Inference with PRS-SIM Fiji plugin</h3>

Given a pre-trained PRS-SIM model and an image or stack to be processed, the Fiji plugin is able to generate the corresponding denoised image or stack. 
The workflow includes following steps:

1. Open the image or stack in Fiji and start PRS-SIM plugin by Clicking `.Plugins > PRS-SIM > predict*`

2. Select the network model file, i.e., .zip file in the format of Save Model bundle. 
Of note, the model file could be trained and saved either by Python codes (see [this gist](https://gist.github.com/asimshankar/000b8d276f211f972168afa138eb3cc7)) or PRS-SIM Fiji plugin, but has to be saved with TensorFlow environment <= 1.15.0.

3. Check inference hyper-parameters. The options and parameters here are primarily selected to perform tiling prediction to save memory of CPUs or GPUs (Number of tiles and Overlap between tiles), and decide whether to show progress dialog or not (Show progress dialog)
	<div align=center><img src="../images/predict.png" alt="Predict Parameter" />
4. Please note that when predicting 3D images, make sure the third dimension of the stack is matched with `0[-1]` dimension of the model:
	<div align=center> <img src="../images/remap.png" alt="Predict Parameter" />
5. Image processing with status bar shown in the message box (if select Show progress dialog).
	<div align=center> <img src="../images/predict_progress.png" alt="UI of Predict" />
6. The denoised output will pop out in separate Fiji windows automatically. Then the processed images or stacks could be viewed, manipulated, and saved via Fiji.

<h3 style="color:white;">Training with PRS-SIM Fiji plugin</h3>
For PRS-SIM model training, we provide realigned images dataset which can directly be used for training, and code to process raw SIM images to realigned images. 
In the latter method, please organize directory structure of your data as below.

>dataset_for_training  
>├─ cell01  
>│    ├─ cell01-view-1.tif  
>│    ├─ cell01-view-2.tif  
>│    ├─ cell01-view-3.tif  
>│    └─ cell01-view-4.tif  
>...


1. Start the plugin by `.Plugins > PRS-SIM > train*` and select the folder containing reailgned images.
	<div align=center> <img src="../images/train.png" alt="Train on augmented data Parameter" />
  
2. Check training hyper-parameters, which include Total number of augmentation, total epochs, iteration number per epoch, batch size and initial learning rate.


3. Click OK to start training. A message box containing training information will pop up.

	<div align=center> <img src="../images/train_progress.png" alt="Train progress"/>

4. Click `Preview` and check the input images and current denoised output images.

	<div align=center> <img src="../images/train_preview.png" alt="Train preview"/>

5. Three types of exit:
	>(i) Press `.Cancel > Close*` to enforce an exit if you don't want to train or save this model.
	>
	>(ii) Press `.Finish Training*` for an early stopping. A window will pop up and you can change the save path and filename.
	>
	>(iii) After the training is completed, A window will pop up and you can change the save path and filename.

	<div align=center> <img src="../images/save_model.png" alt="Save model" />

6. Of note, you can also press *`Export Model*` during training to export the lastest saved model without disposing the training progress.
7. The model saved in training progress can be used in predict progress directly.




<h2 style="color:white;">Reference</h2>
<p>
[1] Müller M, Mönkemöller V, Hennig S, et al. Open-source image reconstruction of super-resolution structured illumination microscopy data in ImageJ[J]. Nature communications, 2016, 7(1): 1-6.<br>
[2] Wen G, Li S, Wang L, et al. High-fidelity structured illumination microscopy by point-spread-function engineering[J]. Light: Science & Applications, 2021, 10(1): 1-12.<br>
[3] Lal A, Shan C, Xi P. Structured illumination microscopy image reconstruction algorithm[J]. IEEE Journal of Selected Topics in Quantum Electronics, 2016, 22(4): 50-63.<br>
[4] Cao R, Li Y, Chen X, et al. Open-3DSIM: an open-source three-dimensional structured illumination microscopy reconstruction platform[J]. Nature Methods, 2023, 20: 1183–1186.<br>
</p>




