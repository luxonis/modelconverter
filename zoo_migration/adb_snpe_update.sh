export SNPE_ROOT=$1
export ARCH=aarch64-oe-linux-gcc11.2
export DSPARCH=hexagon-v73
adb -s 2ab58792 shell "mkdir -p /data/local/tmp/snpe/$ARCH/bin"
adb -s 2ab58792 shell "mkdir -p /data/local/tmp/snpe/$ARCH/lib"
adb -s 2ab58792 shell "mkdir -p /data/local/tmp/snpe/dsp/lib"
adb -s 2ab58792 push $SNPE_ROOT/lib/$ARCH/*.so /data/local/tmp/snpe/$ARCH/lib
adb -s 2ab58792 push $SNPE_ROOT/lib/$DSPARCH/unsigned/*.so /data/local/tmp/snpe/dsp/lib
adb -s 2ab58792 push $SNPE_ROOT/bin/$ARCH/* /data/local/tmp/snpe/$ARCH/bin


echo "export ARCH=aarch64-oe-linux-gcc11.2" > source_me.sh
echo "export ADSP_LIBRARY_PATH=\"/data/local/tmp/snpe/dsp/lib;/dsp;/usr/lib/rfsa/adsp\"" >> source_me.sh
echo "export LD_LIBRARY_PATH=\"/data/local/tmp/snpe/$ARCH/lib\"" >> source_me.sh
echo "export PATH=\"/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/data/local/tmp/snpe/$ARCH/bin\"" >> source_me.sh

adb -s 2ab58792 push source_me.sh /data/local/tmp/
