#!/bin/sh

# This script is downloading MNIST database and unpack it 

echo "Downloading MNIST database..."

# Get MNIST using wget 
wget --no-check-certificate http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget --no-check-certificate http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget --no-check-certificate http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget --no-check-certificate http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

# unpack MNIST files
echo "Unpacking..."
gunzip -v train-images-idx3-ubyte.gz
gunzip -v train-labels-idx1-ubyte.gz
gunzip -v t10k-images-idx3-ubyte.gz
gunzip -v t10k-labels-idx1-ubyte.gz

echo "finished."
