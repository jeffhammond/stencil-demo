#!/bin/bash

MYDIR=/local/${USER}

set +xe

cd $MYDIR
if [ -d ${MYDIR}/DPCPP/.git ] ; then
    cd ${MYDIR}/DPCPP
    git fetch --all
    git checkout sycl
    git pull
    git gc
else
    git clone https://github.com/intel/llvm.git DPCPP
fi
cd ${MYDIR}/DPCPP
python3 ./buildbot/configure.py --cuda
python3 ./buildbot/compile.py
#python3 ./buildbot/check.py
