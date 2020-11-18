#!/bin/bash

cd $HOME
if [ -d ${HOME}/DPCPP ] ; then
    cd ${HOME}/DPCPP
    git fetch --all
    git checkout sycl
    git pull
    git gc
else
    git clone https://github.com/intel/llvm.git DPCPP
fi
cd ${HOME}/DPCPP
python ./buildbot/configure.py --cuda
python ./buildbot/compile.py
#python ./buildbot/check.py
