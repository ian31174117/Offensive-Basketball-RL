import yaml
import os
import shutil
import argparse

def load_yaml(args, yml):
    print("Loading Config from: ", yml, "\n")
    with open(yml, 'r', encoding='utf-8') as fyml:
        dict = yaml.load(fyml.read(), Loader=yaml.Loader)
        for k in dict:
            setattr(args, k, dict[k])
