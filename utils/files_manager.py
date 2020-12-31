import os
import json

def abs_path():
    return os.path.split(os.path.abspath(__file__))[0]

def generate_path(file_name_):
    return os.path.join(abs_path(),file_name_)


def read_config(path):
    "Read configuration file"
    with open(path,encoding='utf-8') as f:
        file = json.load(f)
    return file


def open_file(path):
    "Open Text file by given path"
    with open(generate_path(path)) as f:
        file = f.read()
    return file
