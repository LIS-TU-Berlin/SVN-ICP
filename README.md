<div align="center">
    <h1>SVN-ICP</h1>
    <h3>Uncertainty Estimation of ICP-based LiDAR Odometry using Stein Variational Newton</h3>
    <a href="https://github.com/LIS-TU-Berlin/SVN-ICP"><img src="https://img.shields.io/badge/-C++-blue?logo=cplusplus" /></a>
    <a href="https://github.com/LIS-TU-Berlin/SVN-ICP"><img src="https://img.shields.io/badge/ROS2-Humble-blue" /></a>
    <a href=""><img src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black" /></a>
    <a href="https://github.com/LIS-TU-Berlin/SVN-ICP/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT" /></a>
    <!--a href="https://github.com/LIS-TU-Berlin/SVN-ICP"><img src="https://img.shields.io/badge/DOI-10.1109/xxx.svg"/-->
    <br />
    <br />
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://www.youtube.com/watch?v=CU6aAiTIO6Y">Video</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://arxiv.org/abs/2509.08069">Paper</a>
    <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
    <a href="https://github.com/LIS-TU-Berlin/SVN-ICP/issues">Contact Us</a>
  <br />
  <br />
  <p align="center"><img src=resources/demo.gif alt="animated" width="1800" /></p>
</div>


## :gear: Build

### :package: Dependencies

#### The following dependencies should be installed prior to compile the SVN-ICP workspace.


1. **CUDA** and **LibTorch**

    Install CUDA in your favorit version from <a href="https://developer.nvidia.com/cuda-toolkit">NVIDIA</a>.
    You may want to consulte <a href="https://pytorch.org/get-started/locally/">PyTorch</a> to get which CUDA versions are supported by the current LibTorch. **This is the easiest way to get libtorch installed on your computer!**
    
    If you cannot find the pre-compiled LibTorch from its official site, you can follow <a href="https://github.com/pytorch/pytorch/blob/main/docs/libtorch.rst">THIS INSTRUCTION</a> to build LibTorch for your development environment given a particular CUDA version.

    The current **SVN-ICP** code has been tested with LibTorch 2.5.0+CUDA 12.8 and LibTorch 2.1.6+CUDA 12.6.

2. **ROS2**
    Install the <a href="https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html">ROS2</a> in a distribution **no earilier** than **Humble**. 

3. **GTSAM**
    Compile and install the <a href="https://github.com/borglab/gtsam">GTSAM</a>.

4. **robin-map**
    Compile and install the <a href="https://github.com/Tessil/robin-map">robin-map</a>.


#### :package: Create and Compile the SVN-ICP workspace

    
```sh
mkdir -p ~/svnicp_ws/src
cd ~/svnicp_ws/src
git clone https://github.com/LIS-TU-Berlin/SVN-ICP.git
cd ..
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
source ~/svnicp_ws/install/local_setup.bash
```

You must update the LibTorch path in the `CMakeList.txt` at `line 9`.


**If you ran into errors, please create an issue.**


## :gear: Run

Running SVN-ICP is then straightfoward as


You are also encouraged to have a look into the launch files to check if you want to change the options.


## :sparkles: Contributors

We are also happy to call contributions not only on the code but also for future research collaborations from the robotics community. If you are interested, mail us or open a Pull Request!

<a href="https://github.com/LIS-TU-Berlin/SVN-ICP/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=LIS-TU-Berlin/SVN-ICP" />
</a>


## :floppy_disk: Data

We used two public datasets [**SubT-MRS**][papersubtmrs] and [**GEODE**][papergeode] in our experiments. Both datasets must be and also have been converted in ROS2 bags. 

You can check whether the authors of both datasets provide utilities for this conversion. However, for the convenience of colleagues who are interested in our work, we also provide the ROS2 bags for both datasets at https://drive.google.com/drive/folders/1K3ka9rXKfJ7QkcjnC6uD7CXbaAqxyFg7?usp=sharing. 


**Please cite the datasets correctly if you use them in your research papers:**

[papersubtmrs]: https://arxiv.org/abs/2307.07607
[papergeode]: https://arxiv.org/abs/2409.04961


```
@INPROCEEDINGS{Data_subtmrs,
  author={Zhao, Shibo and Gao, Yuanjun and Wu, Tianhao and Singh, Scherer, Sebastian  and others},
  booktitle={IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)}, 
  title={{SubT-MRS} Dataset: Pushing {SLAM} Towards All-weather Environments}, 
  year={2024},
  doi={10.1109/CVPR52733.2024.02137}}
```
and
```
@article{Data_geode,
author = {Zhiqiang Chen and Yihua Qi and Dapeng Feng and Xuebin Zhuang and Hongbo Chen and others},
title ={Heterogeneous {LiDAR} Dataset for Benchmarking Robust Localization in Diverse Degenerate Scenarios},
journal = {Int. J. Robot. Res. (IJRR)},
year = {2025},
doi = {10.48550/arXiv.2409.04961},
}
```

## :page_with_curl: Citation

To cite our work in your papers, you can use the following bibtex:
```
@article{svn-icp,
author = {Ma, Shiping and Zhang, Haoming and Toussaint, Marc},
title ={{SVN-ICP}: Uncertainty Estimation of {ICP}-based {LiDAR} Odometry using {S}tein {V}ariational {N}ewton},
journal={IEEE Robotics and Automation Letters}, 
year = {2025},
doi = {10.48550/arXiv.2509.08069},
}
```

## :pray: Acknowledgement

Many thanks to the authors of **GenZ-ICP** for this wonderful template of MarkDown.

Please refer to [GenZ-ICP][genzicplink] for more information

[genzicplink]: https://github.com/cocel-postech/genz-icp

## :mailbox: Contact information

If you have any questions, please do not hesitate to contact us

* [Haoming Zhang][hzlink] :e-mail: haoming`dot`zhang `at` rwth-aachen `dot` de
* [Shiping Ma][splink] :e-mail: shiping`dot`ma `at` tu-berlin `dot` de

[hzlink]: https://probablyinconsistent.de/
[splink]: https://github.com/msp666