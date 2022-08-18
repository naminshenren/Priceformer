# Priceformer
Priceformer: An algorithm for forecasting commodity price trend based on Transformer

## Overview
Here we provide the implementation of a  (Priceformer) layer in Pytorch. The repository is organised as follows:
- `dataset/` put you data here ;
- `models/` contains the implementation of the Priceformer network (`Priceformer.py`);
- `checkpoints/` contains a pre-trained Priceformer model;


Finally, `run.py` puts all of the above together and may be used to execute a full training run on you data by executing 
`python3 -u run.py\   
--is_training 1\
--root_path ./dataset/Dataset/\
--data_path **Your data path**\
--model_id *Your model ID*\   
--model Priceformer\
--data *Your data name*\
--features S\
--seq_len 7\   
--label_len 3\
--pred_len 7\
--e_layers 2\
--d_layers 1\
--factor 3\
--enc_in 1\
--dec_in 1\
--c_out 1\
--des 'Exp'\
--itr 1\
--target *Your target*
`.

## Result on donggua Dataset
![image](https://github.com/naminshenren/Priceformer/blob/master/test_results/40.pdf)

## Dependencies

The script has been tested running under Python 3.5.2, with the following packages installed (along with their dependencies):

- `numpy==1.19.2`
- `scipy==1.0.0`
- `scikit-learn==0.24.2`
- `pandas==1.1.5`
- `matplotlib==3.3.4`
- `pytorch==1.4.0`

In addition, CUDA 9.0 and cuDNN 7 have been used.

## Acknowledge
This work was supported by the National Key R&D Program of China under Grant No. 2020AAA0103804(Sponsor: <a  href ="https://bs.ustc.edu.cn/chinese/profile-74.html">Hefu Liu</a>) and partially supported by grants from the National Natural Science Foundation of China (No.72004021). This work belongs to the University of science and technology of China.

