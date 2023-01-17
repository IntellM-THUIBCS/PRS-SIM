---
layout: page
title: Dataset
---

***The dataset will be released once our paper is accepted.***
<br><br>
## Download links
All data used for training and validation of RES-SIM are listed here (public availabile in the future). Experimentally obtained data were captured by our home-built Multi-modality SIM system (2D-SIM, TIRF-SIM, GI-SIM and 3D-SIM)and home-built Lattice-light-sheet SIM system (LLS-SIM).


| No.   |                            Samples                           | Imaging modailty  |      Lables      |  Pixel size  | Frame/volume rate   | Frame/volume number          | FOV                   |     Comments                 |
| :---: | :----------------------------------------------------------: | :---------------: |:---------------: | :----------: | :---------:    | :---------:                  | :-------------------: | :-------------------------- |
|  1   | CCPs in fixed cell<sup>*</sup> | TIRF-SIM  | N.A. | 62.6nm\*62.6nm |       N.A.       |          N.A.           |  31&mu;m\*31&mu;m  | Low-SNR for training and high-SNR for reference |
|  2   | MTs in fixed cell<sup>*</sup>  | TIRF-SIM  | N.A. | 62.6nm\*62.6nm  |       N.A.       |          N.A.          |  31&mu;m\*31&mu;m  | Low-SNR for training and high-SNR for reference |
|  3   | ER in fixed cell<sup>*</sup>   | TIRF-SIM  | N.A. | 62.6nm\*62.6nm  |       N.A.       |          N.A.          |  31&mu;m\*31&mu;m  | Low-SNR for training and high-SNR for reference |
|  4   | MTs in fixed COS-7 cell        | 3D-SIM    | N.A. | 62.6nm\*62.6nm\*160nm |       N.A.       |          N.A.          |  31&mu;m\*31&mu;m\*3.3&mu;m  | Low-SNR for training and high-SNR for reference |
|  5   | Lyso in fixed COS-7 cell       | 3D-SIM    | N.A. | 62.6nm\*62.6nm\*160nm |       N.A.       |          N.A.          |  31&mu;m\*31&mu;m\*2.3&mu;m  | Low-SNR for training and high-SNR for reference |
|  6   | Mito in fixed COS-7 cell       | LLS-SIM   | N.A. | 92.6nm\*92.6nm\*185.2nm |       N.A.       |          N.A.          |  31&mu;m\*31&mu;m\*2.3&mu;m  | Low-SNR for training and high-SNR for reference |
|  7   | ER in fixed COS-7 cell         | LLS-SIM   | N.A. | 92.6nm\*92.6nm\*92.6nm |       N.A.       |          N.A.          |  31&mu;m\*31&mu;m\*2.3&mu;m  | Low-SNR for training and high-SNR for reference |
|  8   | CCPs in live SUM-159 cell         | TIRF-SIM   | N.A. | 62.6nm\*62.6nm\*135nm |       0.5s       |          5000         |  64&mu;m\*64&mu;m | Low-SNR time-lapsed |
|  9   | CCPs and F-actin in live SUM-159 cell         | TIRF-SIM   | N.A. | 62.6nm\*62.6nm\*135nm |       3s       |          170          |  64&mu;m\*64&mu;m | Low-SNR time-lapsed |
|  10   | MTs and lyso in live COS-7 cell         | 3D-SIM   | N.A. | 62.6nm\*62.6nm\*135nm |       10       |          400          |  31&mu;m\*31&mu;m\*33&mu;m  | Low-SNR time-lapsed |
|  11   | Mito and MTs in live COS-7 cell         | LLS-SIM   | N.A. | 62.6nm\*62.6nm\*102nm |      12       |          313          |  31&mu;m\*31&mu;m\*40&mu;m  | Low-SNR time-lapsed |

<!--
Experimentally obtained data were captured by our two-photon imaging system composed of **(1)** a standard two-photon microscope with multi-color detection capability and **(2)** a customized two-photon microscope with two strictly synchronized detection paths. The signal intensity of the high-SNR path is 10-fold higher than that of the low-SNR path. We provide **11 groups of data (~250 GB)**, including synthetic calcium imaging data; recordings of *in vivo* calcium dynamics in mice, zebrafish and flies;  2D and 3D imaging of neutrophil migration; 2D and 3D imaging of mouse cortical ATP release. All data are listed in the table below and we have no restriction on data availability. You can download these data by clicking the **`DOI hyperlinks`** appended in the 'Title' column. We recommend using **[`ImageJ/Fiji`](https://imagej.net/software/fiji/downloads)** to open and view these files.

## Download links

| No.  |                            Title                             |      Events       |  Pixel size  | Frame/volume rate | Imaging Depth<sup>*</sup> | Data size |     Comments      |
| :--: | :----------------------------------------------------------: | :---------------: | :----------: | :---------------: | :-----------------------: | :-------: | :---------------: |
|  1   | <center> Synthetic calcium imaging data<a href="https://doi.org/10.5281/zenodo.6254739"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6254739.svg" alt="DOI"></a></center> | Calcium transient | 1.020 μm/pxl |       30 Hz       |          200 μm           |  29.8 GB  | Low-SNR/high-SNR  |
|  2   | <center> Mouse dendritic spines <a href="https://doi.org/10.5281/zenodo.6275571"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6275571.svg" alt="DOI"></a></center> | Calcium transient | 0.155 μm/pxl |       30 Hz       |           40 μm           |  21.7 GB  | Low-SNR/high-SNR  |
|  3   | <center>Zebrafish optic tectum neurons<a href="https://doi.org/10.5281/zenodo.6339707"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6339707.svg" alt="DOI"></a></center> | Calcium transient | 0.254 μm/pxl |       30 Hz       |            ——             |  6.3 GB  | Low-SNR/high-SNR  |
|  4   | <center> Zebrafish multiple brain regions<a href="https://doi.org/10.5281/zenodo.6293696"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6293696.svg" alt="DOI"></a></center> | Calcium transient | 0.873 μm/pxl |       15 Hz       |            ——             |  7.2 GB  | Low-SNR/high-SNR  |
|  5   | <center><i>Drosophila</i> mushroom body<a href="https://doi.org/10.5281/zenodo.6296555"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6296555.svg" alt="DOI"></a></center> | Calcium transient | 0.254 μm/pxl |       30 Hz       |            ——             |  11.1 GB  | Low-SNR/high-SNR  |
|  6   | <center> Mouse brain neutrophils<a href="https://doi.org/10.5281/zenodo.6296569"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6296569.svg" alt="DOI"></a></center> |  Cell migration   | 0.349 μm/pxl |       10 Hz       |           30 μm           |  11.8 GB  | Low-SNR/high-SNR  |
|  7   | <center> Mouse brain neutrophils (3D)<a href="https://doi.org/10.5281/zenodo.6297924"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6297924.svg" alt="DOI"></a></center> |  Cell migration   | 0.310 μm/pxl |       2 Hz        |  15-45 μm (2 μm/ plane)   |  27.4 GB  | Low-SNR, 2 colors |
|  8   | <center> ATP release in the mouse brain<a href="https://doi.org/10.5281/zenodo.6298010"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6298010.svg" alt="DOI"></a></center> |   ATP dynamics    | 0.465 μm/pxl |       15 Hz       |           20 μm           |  6.0 GB  | Low-SNR/high-SNR  |
|  9   | <center> ATP release in the mouse brain (3D) <a href="https://doi.org/10.5281/zenodo.6298434"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6298434.svg" alt="DOI"></a>  </center> |   ATP dynamics    | 0.698 μm/pxl |       1 Hz        |   10-70 μm (2 μm/plane)   |  50.2 GB  |      Low-SNR      |
|  10  | <center> Mouse neurites <a href="https://doi.org/10.5281/zenodo.6299076"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6299076.svg" alt="DOI"></a></center> | Calcium transient | 0.977 μm/pxl |       30 Hz       |         40-80 μm          |  23.5 GB  | Low-SNR/high-SNR  |
|  11  | <center> Mouse neuronal populations<a href="https://doi.org/10.5281/zenodo.6299096"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.6299096.svg" alt="DOI"></a> </center> | Calcium transient | 0.977 μm/pxl |       30 Hz       |         90-180 μm         |  53.6 GB  | Low-SNR/high-SNR  |

```
*Depth: imaging depth below the brain surface. Only for mouse experiments. 
```

-->


*Data from open-source dataset <a href="https://figshare.com/articles/dataset/BioSR/13264793"> bioSR</a>. For more information about these data, please refer [1].


## Reference
[1]Qiao C, Li D, Guo Y, et al. Evaluation and development of deep neural networks for image super-resolution in optical microscopy[J]. Nature Methods, 2021, 18(2): 194-202.<br>

