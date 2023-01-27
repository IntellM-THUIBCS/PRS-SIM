---
layout: page
title: Tutorial
---

*** We will release the related code and plugin once our manuscirpt is accepted. ***
<h2 style="color:white;">Content</h2>

<ul>
  <li><a href="#Matlab">Matlab code for image re-corruption and SIM reconstruction</a></li>
  <li><a href="#Python">Python code for network training and inference</a></li>
  <li><a href="#Fiji">Fiji plugin</a></li>
</ul>

<h2 style="color:white;" id="Matlab">1. Matlab source code</h2>
<h3 style="color:white;">Create the training dataset</h3>
<p>
The training dataset of RES-SIM is generated with a series of low-SNR raw image stacks (3*3 for a 2D/TIRF-SIM image, 3*5*z_num for a 3D-SIM volume, and 1*3 for a LLS-SIM slice). 
For each raw image stack, we first employ the image re-corruptiont strategy form a corrupted image stack pair, then apply convention SIM algorithm to form a corrupted image (volume) pair,
 and final arrange them as the input data and target data, respectively.
By repeating this operations to each image stack, the final training dataset is formulated. Typically, 30~50 individual image stacks is adequate for a successful training.
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
		<td><font color=white>The name of the smple.</font></td>
	</tr>
	<tr>
		<td><font color=black>File_dir</font></td>
		<td><font color=black>The directory to load the raw data, default is the 'data' folder in current directory.</font></td>
	</tr>
	<tr>
		<td><font color=white>Save_file_dir</font></td>
		<td><font color=white>The directory to save the data, default is the 'data' folder in current directory.</font></td>
	</tr>
	<tr>
		<td><font color=black>Save_file_format</font></td>
		<td><font color=black>The format for saving the re-corrupted data. The format of '.npy' (recommended for fast IO during the training), '.tif', and '.mat' is supported by now. Please note that if the '.tif' format is selected, please make sure the 
		data is save in 'double' mode, otherwise the negative value will be discarded and the denoising performance will be degraded.</font></td>
	</tr>
</table>

<p>
The parameters for SIM modulation:
</p>
<table border="1">
	<tr>
		<td><font color=black>Parameter name</font></td>
		<td><font color=black>Description</font></td>
		<td><font color=black>Typical value <br>(based on our optical system)</font></td>
	</tr>
	<tr>
		<td><font color=white>k0</font></td>
		<td><font color=white>A complex number list representing the illumination pattern in Fourier field. Each invidual number denotes the information for the corresponding direction.
		The absolute value denotes the period of the spatial Moore Fringe (the inverse of its period),
		and the phase value denotes the direction of the pattern. 
		For 2D/TIRF/3D-SIM modality, 3 directions are employed, and for LLS-SIM modality, 1 direction is employed</font></td>
		<td><font color=white>Noted in the code file.</font></td>
	</tr>

	<tr>
		<td><font color=black>mod_factor</font></td>
		<td><font color=black>A complex number list representing the modolation factor for each pattern. Each invidual number denotes the information for the corresponding pattern. 
		The absolute value denotes the modulation depth (the intensity ratio of the corresponding order to 0-order), 
		and the phase value denotes the spatial shift. For 2D/TIRF/LLS-SIM modality, 3 patterns are employed, and for 3D-SIM modality 5 patterns are employed.</font></td>
		<td><font color=black>Noted in the code file.</font></td>
	</tr>
	<tr>
		<td><font color=white>wiener_param</font></td>
		<td><font color=white>Paramamter used for Wiener filter. The proper setting of its value is related to the SNR of the raw images, 
		as the low <b>wiener_param</b> value should be used for low SNR inputs.</font></td>
		<td><font color=white>From 0.001 to 0.05</font></td>
	</tr>
	<tr>
		<td><font color=black>otf_path</font></td>
		<td><font color=black>The path where the otf file is saved. For commercial SIM system, the otf file should be located in the configuration folder. 
		If it is not available, it can also be estimated by open-source package <a href="https://www.fairsim.org/">fairSIM<sup>[1]</sup></a>.</font></td>
		<td><font color=black>-</font></td>
	</tr>
	<tr>
		<td><font color=white>NA_em</font></td>
		<td><font color=white>The effective detection NA of the microscopy system. This parameter is used for precise estimation of the system OTF.</font></td>
		<td><font color=white>1.3</font></td>
	</tr>
	<tr>
		<td><font color=black>lambda_em</font></td>
		<td><font color=black>The wavelength of the emission light (in nm). This parameter is used for precise estimation of the system OTF.</font></td>
		<td><font color=black>525/609/705</font></td>
	</tr>
	<tr>
		<td><font color=white>n_imm</font></td>
		<td><font color=white>The refractive index of the immersion oil. This parameter is used for precise estimation of the system OTF.</font></td>
		<td><font color=white>1.78 for TIRF-SIM, 1.51 for 3D-SIM and 0.5 for LLS-SIM</font></td>
	</tr>
</table>

<p>
The parameters for image re-corruption:
</p>
<table border="1">
	<tr>
		<td><font color=black>Parameter name</font></td>
		<td><font color=black>Description</font></td>
		<td><font color=black>Typical value range ([min, max])</font></td>
	</tr>
	<tr>
		<td><font color=white>&alpha;</font></td>
		<td><font color=white>Noise control factor: to adjust the re-corruption intensity</font></td>
		<td><font color=white>[2, 5]</font></td>
	</tr>
	<tr>
		<td><font color=black>&beta;<sub>1</sub></font></td>
		<td><font color=black>Poisson factor: to represent the intensity of signal-related noise (e.g. shot noise)</font></td>
		<td><font color=black>[0.8, 3]</font></td>		
	</tr>
	<tr>
		<td><font color=white>&beta;<sub>2</sub></font></td>
		<td><font color=white>Gaussian factor: to represent the intensity of Gaussian white noise (e.g. readout noise)</font></td>
		<td><font color=white>[0.4, 6]</font></td>
	</tr>
	<tr>
		<td><font color=black>repeat_time</font></td>
		<td><font color=black>The number of re-corruption operations applied for each image stack. (For the purpose of enriching the training dataset)</font></td>
		<td><font color=black>3</font></td>
	</tr>
</table>

<h3 style="color:white;">Pre-process the noisy data for inference</h3>
<p>
During the inference phase, the noisy raw data is first reconstructed by conventional SIM algorithm to generated the noisy SIM SR images (volumes), then fed into the pre-trained model to output the final denoised images (volumes).
To generated the data for network input, please run the code in Matlab command window as:<br>
</p>
<code style="background-color:#393939;">
run('SIM_reconstruction.m');
</code><br>
<p>
Similar to create the training dataset, the file Io parameters and the SIM modulation parameter need to be assigned in the code. <br>
The conventional SIM reconstruction algorithm can also be replaced with your own code,
 e.g. the open-source package fairSIM<sup>[1]</sup>, Hifi-SIM<sup>[2]</sup>, and OpenSIM<sup>[3]</sup>.
</p>

<h2 style="color:white;" id="Python">2. Python source code</h2>

<p>
The python code is set for network training and denoising implementation. To accelerate the training process, a GPU equipped computer is required.
</p>

<h3 style="color:white;">Environment installation</h3>

<p>
To avoid the version incompatiblility, we highly recommend to install RES-SIM in a CONDA vitual environment as 
</p>
<code style="background-color:#393939;">
conda create -n ressim python=3.7
</code><br>
<p>
The packages required for RES-SIM are listed in file 'requirements.txt', which can be installed by pip or conda package management command. 
Pleased note that the suggested version information is only validated on our working station, and possibly need to be changed to adjust your specific environment. 
</p>
<code style="background-color:#393939;">
conda install pytorch torchvision torchaudio pytorch-cuda==11.6 -c pytorch -c nvidia<br>
pip install tifffile<br>
pip install tensorboardX<br>
</code><br>

<p>
To begin the training or inference processing, just activate the vitual environment and the excecute the corresponding code.
</p>
<code style="background-color:#393939;">
cd ./<br>
conda activate ressim<br>
</code><br>

<h3 style="color:white;">Network training</h3>
<p>
Before the training, we need to organize the folder saving the training dataset in the following mode (if the dataset is generated with aforementioned Matlab code, this requirement will be already satisfied):
</p>
* root_dir
   * input       --- to save the input data for training
   * target      --- to save the target data for training
   * val_raw     --- to save the input data for validation
   * infer_raw   --- to save the input data for denoising inference
<br>
<p>
Please note that the files in *input* folder and *target* folder should be strictly paired (We highly recommended to name the paired files with the same name as '000001.npy' to avoid the mismatching), 
otherwise a warning window will appear. If your want to change the dataset archetecture, please re-write the related data IO code in 'dataset.py'. <br>
</p>

<p>
Our code is developed based on open source architecture KAIR (<a href="https://github.com/cszn/KAIR">https://github.com/cszn/KAIR</a>), in which the interface for multiple network, model and dataset is provided. 
The training condition is defined in the '.json' file located in folder 'options'. To change the training condition, just modify the parameters in the corresponding file. 
The main tubable parameters for training are listed in the following table, and other customized parameters can be added for the specific use. <br>
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
If your want to employed other network, 
just created a new '.py' file for network and a '.json' file for training condition, and then create a link by adding a selection in 'select_model.py'. <br>
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
		<td><font color=black>load_model_iter</font></td>
		<td><font color=black>int</font></td>
		<td><font color=black>The iteration of the pre-trained model to load, default is 0 (to train a new model) </font></td>
	</tr>
</table>

<p>
The example code for training is :
</p>

<code style="background-color:#393939;">
cd ./<br>
python Main_train.py --smpl_dir dataset --smpl_name microtubules --data_format npy  --network_type unet --gpu_id 0 --save_suffix _1  --preload_data_flag --load_model_iter 0<br>
</code><br>

<h3 style="color:white;">Denoising</h3>

<p>
To implement the denoising with the pre-trained model, just run the following command:
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
		<td><font color=black>model_suffix</font></td>
		<td><font color=black>str</font></td>
		<td><font color=black>The suffix of the pre-trained model</font></td>
	</tr>
	<tr>
		<td><font color=white>load_model_iter</font></td>
		<td><font color=white>int</font></td>
		<td><font color=white>The iteration of the pre-trained model to load, default is 0 (to train a new model) </font></td>
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
python Main_test.py --smpl_dir data --smpl_name Microtubules --model_suffix _1 --data_format npy --network_type unet --gpu_id 0 --load_model_iter -1 
</code><br>

<h3 style="color:white;">Examples</h3>
<p>
A representative training dataset and inference dataset are located in the 'dataset' of current repository. To provide a intuitive view of our software, 
we also uploaded the corresponding pre-trained for this dataset. The example training image and denoised are shown in the following figure.
</p>

<img src="../images/Demo-website-training.png?raw=true" width="600" align="middle">

<h2 style="color:white;" id="Fiji">3. Fiji plugin</h2>
<p>
To make our work convenient for more biological researchers, we also provided a ready-to-use Fiji plugin for SIM denoising. The detailed instruction of our plugin is provided in the following section.<br>
</p>

<!--
The layout of our plugin is demostrated in Figure 2. The noisy SIM SR image (after reconstruction) and the pre-trained model are needed.
The SR image can be reconstructed by either the open-source software (e.g. fairSIM) or home-built code. 
To further improve the convenience, the interface between fairSIM and our plugin are under developing. 
To perform the denoising, just open the noisy images and designate the path of the model package, then the denoise SR images will be displayed in a new window.
To validate our plugin, the example data of different type of organlles and their corresponding models have been uploaded in the repository. 

<center> Figure 2 | The layout and example of our Fiji plugin.</center>
-->
<h3 style="color:white;">Installation</h3>

1. Copy all the files in folder `./jars` `./plugins` to the corresponding folder in `/$YourPath/Fiji.app/`
2. Restart Fiji
3. Access to RES-SIM plugin is as "Fiji menu -> plugins -> RES-SIM":

   !["Access to RES-SIM Fiji Plugin"](../images/access.png "Access to RES-SIM Fiji Plugin")

<h3 style="color:white;">Denoising with RES-SIM plugin</h3>

1. Open Fiji.
2. Open/Choose an image.
3. Find the corresponding model in './models', or you can export your model as <b>ZIP file</b> following
[this gist](https://gist.github.com/asimshankar/000b8d276f211f972168afa138eb3cc7) in python code 
and Tensorflow <= 1.15.0 environment.
4. Run the plugin via `Plugins > RES-SIM > Predict`.
5. Designate the following parameters for predicting image:

	![Predict Parameter](../images/predict.png "Predict Parameter")
   
	+ <b>Number of tiles</b>: Part the large input image to several small images while predicting, to avoid out of memory error, and to accelerate the progress of predicting.
	+ <b>Overlap between tiles</b>: The percentage of the pixels in the edge overlapped between the adjacent tiles.
	 Since the edge regions in each tile is lack of effection information, less overlapped ratio will introduce more stitching seam artifact.
	<u>The typically setting of the overlapped ration of 32% is adequate to mininize this effect.</u>
	+ <b>Batch size</b>: The number of image predicted at one time, to accelerate the progress of predicting for large GPU/CPU memory.
	+ <b>Import model (.zip)</b>: Press `Browse` to load the saved model.
	+ <b>Adjust mapping of TF network input</b>: Press `Adjust mapping of TF network input` button,
	 and adjust the match of dimensions of image and dimensions of input in the model.
	 <u>When using RES-SIM 3D model, please check the thirddimension of image is matched with `3[153]` dimension of model.</u> 
	+ <b>Show progress dialog</b>: Show log information in predict progress.

After the aformentioned setting is done, by pressing `Ok`, the predicting progress will begin and the following UI will be shown:<br>
	![UI of Predict](../images/predict progress.png "UI of Predict")

<h3 style="color:white;">Training with RES-SIM plugin</h3>

1. Open input and target data as two separated stack files with Fiji.
2. Run the plugin via `Plugin > RES-SIM > train`.
3. Adjust the following parameters for training:

    ![Train Parameter](../images/train.png "Train Parameter")

    + <b>Input image for training</b>: Choose the input images of dataset.
    + <b>GT image for training</b>: Choose the ground truth images of dataset.
    + <b>Total epochs</b>: How many times of dataset performing during training. 
    + <b>Iteration number per epoch</b>: Iteration number of model in one times of dataset performed during training. 
    + <b>Batch size</b>: the number of input images batched in one iteration of model. 
    + <b>Initial learning rate</b>: The learning rate at start of training.

6. Run the plugin by pressing `Ok`.

7. During Training, You can track the training progress via UI showed below:
   1. Train preview window shows the input image and output image of the current 
   training status.
   
      ![Train preview](../images/train preview.png "Train preview") 
   2. You can see the training loss and validation loss ploted below.
      ![Train progress](../images/train progress.png "Train progress")
      + Press `Cancel > Close` to dispose training progress.
      + Press `Finish` to finish training progress and save model following instructions below.
      + Press `Export Model` to export the lastest save model at current status following instructions below without disposing training progress.
   3. When pressing `Finish` or `Export Model`, you will see the information of the saved model.
      1. In `Overview` `Metadata` `inputs & outputs` `Training`, you will see the parameters of trained model.
      2. Press `File actions > Save to..` to save the file of trained model.
      3. Extract the file of trained model, `./tf_saved_model_bundle.zip` is used for model file(.zip) in `Plugin >
         Plugin > RES-SIM > Predict`.
   
      ![Save model](../images/save model.png "Save model")
   
***

<h3 style="color:white;">Switching the TensorFlow version</h3>
By default, RES-SIM Fiji Plugin ships with TensorFlow 1.15.0 which is compatible to CUDA 10.0 and cuDNN >= 7.4.1.
For supporting a model trained with a specific TensorFlow version or for GPU support:

1. Open `Edit > Options > TensorFlow...`
2. Choose the version matching your system / model. 
3. For GPU support, install CUDA and cuDNN matching the TensorFlow version you choose, and make sure Fiji knows about 
the installation paths.
4. Wait until a message opens telling you that the library was installed. 
5. Restart Fiji.

  ![Edit > Option > Tensorflow](../images/tensorflow.png)


<p>
<b><i>More functions, e.g. the 3D model training with RES-SIM plugin, is under developing, and will be released in the further.</i></b>
</p>


<h2 style="color:white;">Reference</h2>
<p>
[1] Müller M, Mönkemöller V, Hennig S, et al. Open-source image reconstruction of super-resolution structured illumination microscopy data in ImageJ[J]. Nature communications, 2016, 7(1): 1-6.<br>
[2] Wen G, Li S, Wang L, et al. High-fidelity structured illumination microscopy by point-spread-function engineering[J]. Light: Science & Applications, 2021, 10(1): 1-12.<br>
[3] Lal A, Shan C, Xi P. Structured illumination microscopy image reconstruction algorithm[J]. IEEE Journal of Selected Topics in Quantum Electronics, 2016, 22(4): 50-63.<br>
</p>




<!--
## Content

- [1. Python source code](#1-python-source-code)
- [2. Jupyter notebook](#2-jupyter-notebook)
- [3. Colab notebook](#3-colab-notebook)
- [4. Matlab implementation for real-time processing](#4-matlab-implementation-for-real-time-processing)

## 1. Python source code

### UPDATE v0.7 (June 2022) 

We replaced 12-fold data augmentation with 16-fold data augmentation for more stable results. 

Denoising performance (SNR) with the increase of training epochs on simulatedc calcium imaging data:
<center><img src="https://github.com/cabooster/DeepCAD-RT/blob/main/images/16aug.png?raw=true" width="600" align="middle" /></center>

### 1.1 Our environment 

* Ubuntu 16.04 
* Python 3.6
* Pytorch 1.8.0
* NVIDIA GPU (GeForce RTX 3090) + CUDA (11.1)

### 1.2 Environment configuration

1. Create a virtual environment and install PyTorch. In the 3rd step, please select the correct Pytorch version that matches your CUDA version from [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/). 

   ```
   $ conda create -n deepcadrt python=3.6
   $ conda activate deepcadrt
   $ pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
   ```

2. We made a installable pip release of DeepCAD-RT [[pypi](https://pypi.org/project/deepcad/)]. You can install it by entering the following command:

   ```
   $ pip install deepcad
   ```

### 1.3 Download the source code

```
$ git clone https://github.com/cabooster/DeepCAD-RT
$ cd DeepCAD-RT/DeepCAD_RT_pytorch/
```

### 1.4 Demos

To try out the Python code, please activate the `deepcadrt` environment first:

```
$ conda activate deepcadrt
$ cd DeepCAD-RT/DeepCAD_RT_pytorch/
```

**Example training**

To train a DeepCAD-RT model, we recommend starting with the demo script `demo_train_pipeline.py`. One demo dataset will be downloaded to the `DeepCAD_RT_pytorch/datasets` folder automatically. You can also download other data from [the companion webpage](https://cabooster.github.io/DeepCAD-RT/Datasets/) or use your own data by changing the training parameter `datasets_path`. 


```
python demo_train_pipeline.py
```

**Example testing**

To test the denoising performance with pre-trained models, you can run the demo script `demo_test_pipeline.py` . A demo dataset and its denoising model will be automatically downloaded to `DeepCAD_RT_pytorch/datasets` and `DeepCAD_RT_pytorch/pth`, respectively. You can change the dataset and the model by changing the parameters `datasets_path` and `denoise_model`.

```
python demo_test_pipeline.py
```

## 2. Jupyter notebook

We provide simple and user-friendly Jupyter notebooks to implement DeepCAD-RT. They are in the `DeepCAD_RT_pytorch/notebooks` folder. Before you launch the notebooks, please configure an environment following the instruction in [Environment configuration](#12-environment-configuration) . And then, you can launch the notebooks through the following commands:

```
$ conda activate deepcadrt
$ cd DeepCAD-RT/DeepCAD_RT_pytorch/notebooks
$ jupyter notebook
```

<center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/deepcad8.png?raw=true" width="900" align="middle"></center> 

## 3. Colab notebook

We also provide a cloud-based notebook implemented with Google Colab. You can run DeepCAD-RT directly in your browser using a cloud GPU without configuring the environment. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cabooster/DeepCAD-RT/blob/main/DeepCAD_RT_pytorch/notebooks/DeepCAD_RT_demo_colab.ipynb)

*Note: Colab notebook needs longer time to train and test because of the limited GPU performance offered by Colab.*

<center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/deepcad7.png?raw=true" width="700" align="middle"></center> 

## 4. Matlab implementation for real-time processing

To achieve real-time denoising, DeepCAD-RT was optimally deployed on GPU using TensorRT (Nvidia) for further acceleration and memory reduction. We also designed a sophisticated time schedule for multi-thread processing. Based on a two-photon microscope, real-time denoising has been achieved with our Matlab GUI of DeepCAD-RT (tested on a Windows desktop with Intel i9 CPU and 128 GB RAM).

<center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/GUI2.png?raw=true" width="950" align="middle"></center> 



### 4.1 Required environment

- Windows 10
- CUDA 11.0
- CUDNN 8.0.5
- Matlab 2018a (or newer version)
- Visual Studio 2017

### 4.2 File description

`deepcad_trt.m`: Matlab script that calls fast processing and tiff saving function programmed in C++

`deepcad_trt_nosave.m`: Matlab script that calls fast processing function programmed in C++ and save tiff in Matlab

`realtime_core.m`: Realtime simulation in Matlab & C++ and save tiff

`DeepCAD-RT-v2.x.x/DeepCAD-RT-v2/deepcad/+deepcadSession`: Real-time inference with data flow from ScanImage

`DeepCAD-RT-v2.x.x/DeepCAD-RT-v2/results`: Path to save result images

`DeepCAD-RT-v2.x.x/DeepCAD-RT-v2/engine_file`: Path for the engine file

### 4.3 Instructions for use

#### Install

1. Download the `.exe` file from [here](https://doi.org/10.5281/zenodo.6352526). When you double click this self-extracting file, the relevant files of DeepCAD-RT will unzip to the location that you choose.

2. Copy the `.dll` files from `<installpath>/DeepCAD-RT-v2.x.x/dll` to your CUDA installation directory, for example `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin`. The CUDA installer should have already added the CUDA path to your system PATH (from [TensorRT installation guide](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-601/tensorrt-install-guide/index.html#installing-zip)).

#### Model preparation

   After [training](https://github.com/cabooster/DeepCAD-RT#demos), the ONNX files will be saved in `DeepCAD-RT/DeepCAD_RT_pytorch/onnx`. In order to reduce latency, `patch_t` should be decreased. **The recommended training patch size is 200x200x40 pixels.**

   We provide two pre-trained ONNX models in `DeepCAD-RT-v2.x.x/DeepCAD-RT-v2` . The patch size of `cal_mouse_mean_200_40_full.onnx` and `cal_mouse_mean_200_80_full.onnx` are 200x200x40 pixels and 200x200x80 pixels, respectively. The calcium imaging data used for training these model were captured by our customized two-photon microscope.


#### Realtime inference with ScanImage

<center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/GUI.png?raw=true" width="600" align="middle"></center> 

Matlab configuration:

1. Open Matlab.

2. Change file path to `<installpath>/DeepCAD-RT-v2.x.x/DeepCAD-RT-v2`.

3. Configure the environment:

   ```
   mex -setup C++
   
   installDeepCADRT
   ```

4. Open ScanImage and DeepCAD_RT GUI:

   ```
   scanimage
   
   DeepCAD_RT 
   ```

5. Set the parameters in GUI:

   `Model file`: The path of the ONNX file.  Click `...` to open the file browser and choose the file used for inference.

   `Save path`:The path to save denoised images. Click `...` to open the file browser and choose the path

   `Frames number`: How many frames to acquire. It is equal to the value set in ScanImage. This parameter will update automatically when you click `Configure`. 

   <center><img src="https://github.com/cabooster/DeepCAD-RT/blob/page/images/scanimage_parameter.png?raw=true" width="250" align="middle"></center>

   **Attention: You should set the frame number before clicking `Configure`.**

   `Display setting`: 

   `Manual` mode: You can set the minimum and maximum intensity for image display.

   `Auto` mode: The contrast will be set automatically but slightly slower than `Manual` mode.

   `Advanced`: Advanced settings.

   `Patch size (x,y,t)`: The three parameters depend on the patch size you set when you convert Pytorch model to ONNX model.

   `Overlap factor`: The overlap factor between two adjacent patches. The recommended number is between 0.125 and 0.4. Larger overlap factor means better performance but lower inference speed.

   `Input batch size`: The number of frames per batch. The recommended value is between 50 and 100. It should be larger than the patch size in t dimension.

   `Overlap frames between batches`: The number of overlapping slices between two adjacent batches. The recommended value is between 5 and 10. More overlapping frames lead to better performance but lower inference speed.

6. After set all parameters, please click `Configure`. If you click `Configure` for the first time, the initialization program will execute automatically.

7. You can click `GRAB` in ScanImage and start imaging.

8. Before each imaging session, you should click  `Configure`.


### GUI demo

<center><iframe width="800" height="500" src="https://www.youtube.com/embed/u1ejSaVvWiY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe> </center>

-->
