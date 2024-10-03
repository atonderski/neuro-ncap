#!/usr/bin/env bash

DEFUALT_DIR=$PWD/../neurad-studio/checkpoints
DEFUALT_DIR=$(realpath $DEFUALT_DIR)
# ask user where they want to RESOLVED_DIR
read -p "Where would you like to install the neurad-studio checkpoints? (default: $DEFUALT_DIR): " INSTALL_DIR


# if user does not provide a path, use default
if [ -z $INSTALL_DIR ]; then
    INSTALL_DIR=$DEFUALT_DIR
fi

# if folder does not exist, create it
if [ ! -d $INSTALL_DIR ]; then
    mkdir -p $INSTALL_DIR
    echo "Created directory $INSTALL_DIR"
fi

# download the weights
wget -O ncap_weights.tar.gz 'https://www.dropbox.com/scl/fi/s5wu0jwhhmtafmnl8icao/ncap-weights.tar.gz?rlkey=xrf7b99q4z1a09g2mbfl921kk&st=dil5h0m6&dl=0'
# extract the weights
tar -xvf ncap_weights.tar.gz -C $INSTALL_DIR
# remove the tar file
rm ncap_weights.tar.gz
echo "Weights downloaded and extracted to $INSTALL_DIR"
