DiffRim: A Diffusion-Driven Model for High Efficiency Radar Interference Mitigation (Submitted to ICASSP 2026)

1-Data generation

The large-scale synthetic dataset based on RaDICaL [1] and simulated interference can be downloaded from https://pan.baidu.com/s/1dJqVG10XXAzbp_NEFFzSow?pwd=iwm9 (code: iwm9), or generated following the steps in folder '1-Data_generation'. 
‘radar_raw_cube’ in the MATLAB script is referred to as Radar raw ADC data extracted from RaDICaL .bag files.

markdown
```matlab
MATLAB run Radical_inf_adding.m
```

2-Train the model

First, use preprocessing.py to combine 3 consecutive RD frames with an overlap 2 and split into train, val and test sets. 
markdown
```python
python preprocessing.py   # with your data path.
```

Secondly, train and test model with only_test=False,
markdown
```python
python train_rddm.py
```

or, only test model setting only_test True.

Additionally, a pre-trained model can be downloaded from folder 'pre-trained'.


```bibtex
 @article{lim2021radical,
  title={Radical: A synchronized fmcw radar, depth, imu and rgb camera data dataset with low-level fmcw radar signals},
  author={Lim, Teck-Yian and Markowitz, Spencer A and Do, Minh N},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  volume={15},
  number={4},
  pages={941--953},
  year={2021},
  publisher={IEEE}
 }
```
