#!/bin/bash

python3 preprocess/create_mask.py

python3 preprocess/apply_mask.py

python3 preprocess/apply_register.py
