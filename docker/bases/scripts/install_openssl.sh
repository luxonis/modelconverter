#!/bin/bash
set -e

apt-get install wget make -y
wget https://www.openssl.org/source/openssl-1.1.1b.tar.gz
mkdir /opt/openssl
tar xfvz openssl-1.1.1b.tar.gz --directory /opt/openssl
echo /opt/openssl/lib > /etc/ld.so.conf.d/openssl-1.1.1b.conf
cd /opt/openssl/openssl-1.1.1b
./config --prefix=/opt/openssl --openssldir=/opt/openssl/ssl
make
make install
