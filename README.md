# ENFformer: Long-Short Term Representation of Electric Network Frequency for Digital Audio Tampering Detection

## Citations
### APA
[1] Zeng, C., Li, K., & Wang, Z. (2024). ENFformer: Long-short term representation of electric network frequency for digital audio tampering detection. Knowledge-Based Systems, 297, 111938. https://doi.org/10.1016/j.knosys.2024.111938

[2] Zeng, C., Kong, S., Wang, Z., Feng, S., Zhao, N., & Wang, J. (2024). Deletion and insertion tampering detection for speech authentication based on fluctuating super vector of electrical network frequency. Speech Communication, 158, 103046. https://doi.org/10.1016/j.specom.2024.103046

[3] Zeng, C., Kong, S., Wang, Z., Li, K., Zhao, Y., Wan, X., & Chen, Y. (2024). Discriminative Component Analysis Enhanced Feature Fusion of Electrical Network Frequency for Digital Audio Tampering Detection. Circuits, Systems, and Signal Processing. https://doi.org/10.1007/s00034-024-02787-y

### BibTex
@article{Zeng2024a,
  title = {ENFformer: Long-Short Term Representation of Electric Network Frequency for Digital Audio Tampering Detection},
  shorttitle = {ENFformer},
  author = {Zeng, Chunyan and Li, Kun and Wang, Zhifeng},
  year = {2024},
  month = aug,
  journal = {Knowledge-Based Systems},
  volume = {297},
  pages = {111938},
  issn = {0950-7051},
  doi = {10.1016/j.knosys.2024.111938},
  urldate = {2024-05-28}
}

@article{Zeng2024b,
  title = {Deletion and Insertion Tampering Detection for Speech Authentication Based on Fluctuating Super Vector of Electrical Network Frequency},
  author = {Zeng, Chunyan and Kong, Shuai and Wang, Zhifeng and Feng, Shixiong and Zhao, Nan and Wang, Juan},
  year = {2024},
  month = mar,
  journal = {Speech Communication},
  volume = {158},
  pages = {103046},
  issn = {0167-6393},
  doi = {10.1016/j.specom.2024.103046},
  urldate = {2024-02-27}
}

@article{Zeng2024c,
  title = {Discriminative Component Analysis Enhanced Feature Fusion of Electrical Network Frequency for Digital Audio Tampering Detection},
  author = {Zeng, Chunyan and Kong, Shuai and Wang, Zhifeng and Li, Kun and Zhao, Yuhao and Wan, Xiangkui and Chen, Yunfan},
  year = {2024},
  month = jul,
  journal = {Circuits, Systems, and Signal Processing},
  issn = {1531-5878},
  doi = {10.1007/s00034-024-02787-y},
  urldate = {2024-08-17},
  langid = {english}
}


## Introduction 
This is an implementation of the ENFformer model referring to the following paper: ENFformer: Long-Short Term Representation of Electric Network Frequency for Digital
Audio Tampering Detection

## Contributors
1. Kun Li : 102210257@hbut.edu.cn
2. Zhifeng Wang : zfwang@ccnu.edu.cn</br>

School of Electrical and Electronic Engineering, Hubei University of Technology, Wuhan 430068, China

## Environment Requirement
python == 3.7</br>
tensorflow == 2.6.0</br>
keras == 2.6.0</br>
numpy == 1.19.5</br>
scikit-learn == 1.0.2</br>
librosa == 0.8.0</br>

## Datasets
1. The shallow feature file [F01H500next_fram_len_148_256_148.txt] of the Carioca dataset is put into `./Feature_data`, containing the zero-order phase and first-order phase features with a size of 25x83 and the frequency feature with a size of 256x148.

2. The shallow feature file [F01H2000next_fram_len_148_256_148.txt] of the ENF-EDIT1 dataset is put into `./Feature_data`, containing the zero-order phase and first-order phase features with a size of 25x83 and the frequency feature with a size of 256x148.

3. The shallow feature file [F01H5168next_fram_len_148_256_148.txt] of the ENF-EDIT2 dataset is put into `./Feature_data`, containing the zero-order phase and first-order phase features with a size of 25x83 and the frequency feature with a size of 256x148.

## Train and test 
Run the `ENFformer_main.py` to train and test.

The trained models will be saved in file `./logdir0_83_25`. Then call the training model to complete the test, and the top 10 training models with the highest ACC will be re-saved to file `./model0pad_mode_25_83`. Finally, the ten models are saved in the name format [model_epochNUM1_valaccNUM2.hdf5], where NUM1 is the epoch, NUM2 is the ACC, and the highest ACC is selected as the final test accuracy.
