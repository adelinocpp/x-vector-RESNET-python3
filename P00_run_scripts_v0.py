#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 07:37:12 2021

@author: adelino
"""
import os
import subprocess
from glob import glob

def find_sript(directory, pattern='P*.py'):
    return glob(os.path.join(directory, pattern), recursive=True)

files = glob('./LDA_saved/*')
for f in files:
    os.remove(f)

list_script = find_sript("./");
list_script.sort()
first_script = 3



string_command = []
for index, program in enumerate(list_script):
    if (index < first_script):
        continue
    string_command.append("{:}".format(program[2:]))

for command in string_command:
    p = subprocess.call(['python3',command]) # something long running

#subprocess.call(string_command)

