# Paired Variational Autoencoders

[Embodied Language Learning with Paired Variational Autoencoders](https://ieeexplore.ieee.org/document/9515668)

Last updated: 7 May 2024.

This code has been partially adapted from Copyright (c) 2018, [Tatsuro Yamada](https://github.com/ogata-lab/PRAE/)

Copyright (c) 2021, Ozan Özdemir <<ozan.oezdemir@uni-hamburg.de>>

## Requirements
- Python 3.6
- Tensorflow 1.4
- NumPy 1.19

## Implementation
Paired Variational Autoencoders

## Example
```
$ cd src
$ python learn.py
```
- learn_pvae.py: trains the model.
- generation.py: translates instructions to actions.
- recognition.py: translates actions to descriptions.
- extraction.py: extracts shared representations.
- reproduction.py: reproduces the actions.

## Citation

**PVAE**
```bibtex
@InProceedings{OKW21, 
 	 author =  {Özdemir, Ozan and Kerzel, Matthias and Wermter, Stefan},  
 	 title = {Embodied Language Learning with Paired Variational Autoencoders}, 
 	 booktitle = {2021 IEEE International Conference on Development and Learning (ICDL)},
 	 number = {},
 	 volume = {},
 	 pages = {1--6},
 	 year = {2021},
 	 month = {Aug},
 	 publisher = {IEEE},
 	 doi = {10.1109/ICDL49984.2021.9515668}
 }
```
