#! /bin/bash
dataset=$1
path=$2
mkdir -p $path
cd $path

if [[ $dataset = "py" ]]
then
    wget http://files.srl.inf.ethz.ch/data/py150.tar.gz
    tar -zxf py150.tar.gz
    wget http://files.srl.inf.ethz.ch/data/py150_files.tar.gz
    tar -zxf py150_files.tar.gz
fi

if [[ $dataset = "js" ]]
then
    wget http://files.srl.inf.ethz.ch/data/js_dataset.tar.gz
    tar -zxf js_dataset.tar.gz
fi