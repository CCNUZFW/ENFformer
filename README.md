# ENFformer: Long-Short Term Representation of Electric Network Frequency for Digital Audio Tampering Detection

## About 
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