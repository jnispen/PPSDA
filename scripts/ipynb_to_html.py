#!/usr/bin/python

import sys
import os
import subprocess

if len(sys.argv) < 2:
    print("Please provide a search path..")
    sys.exit()

""" search for .ipynb files in the search path and convert to html """
dir_path = os.path.dirname(os.path.realpath(__file__))
search_dir = dir_path + "/" + sys.argv[1]
for root, dirs, files in os.walk(search_dir):
    # exclude the '.ipynb_checkpoints' dirs
    dirs[:] = [d for d in dirs if not '.ipynb_checkpoints' in d]
    for file in files:
        if file.endswith('.ipynb'):
            file_in = root + "/" + str(file)
            file_out = "html/" + os.path.splitext(str(file))[0] + ".html"
            #print(file_in)
            #print(file_out)
            cmd = ['jupyter', 'nbconvert', file_in, '--to', 'html', '--output', file_out]
            #print(cmd)
            subprocess.call(cmd)
