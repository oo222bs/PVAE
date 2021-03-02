# Paired Variational Autoencoders

Paired Variational Autoencoders

Last updated: 2 March, 2021.

This code has been adapted from Copyright (c) 2018, Tatsuro Yamada <<yamadat@idr.ias.sci.waseda.ac.jp>>

Original repository: https://github.com/ogata-lab/PRAE/

Copyright (c) 2021, Ozan Özdemir <<oezdemir@informatik.uni-hamburg.de>>

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