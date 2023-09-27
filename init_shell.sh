#!/bin/bash

pip install wandb
pip install transformers
pip install peft
pip install sentencepiece
pip install datasets
pip install bitsandbytes
pip install accelerate==0.22.0
pip install fire
pip install -r requirements.txt -q
pip uninstall peft -y -q
pip install -q git+https://github.com/huggingface/peft.git@e536616888d51b453ed354a6f1e243fecb02ea08

#/content/drive/MyDrive/Colab/NewsRec/shell_script/init_colab_shell.sh