set -e

export SNPE_ROOT=$1
export ARCH=aarch64-oe-linux-gcc11.2
export DSPARCH=hexagon-v73

ADB_ROOT=/data/local/tmp/snpe_2_32_6

adb -s 2ab58792 shell "mkdir -p $ADB_ROOT/$ARCH/bin"
adb -s 2ab58792 shell "mkdir -p $ADB_ROOT/$ARCH/lib"
adb -s 2ab58792 shell "mkdir -p $ADB_ROOT/dsp/lib"
adb -s 2ab58792 push $SNPE_ROOT/lib/$ARCH/*.so $ADB_ROOT/$ARCH/lib
adb -s 2ab58792 push $SNPE_ROOT/lib/$DSPARCH/unsigned/*.so $ADB_ROOT/dsp/lib
adb -s 2ab58792 push $SNPE_ROOT/bin/$ARCH/* $ADB_ROOT/$ARCH/bin


echo "export ARCH=aarch64-oe-linux-gcc11.2" > source_me.sh
echo "export ADSP_LIBRARY_PATH=\"$ADB_ROOT/dsp/lib;/dsp;/usr/lib/rfsa/adsp\"" >> source_me.sh
echo "export LD_LIBRARY_PATH=\"$ADB_ROOT/$ARCH/lib\"" >> source_me.sh
echo "export PATH=\"$ADB_ROOT/$ARCH/bin\":$PATH" >> source_me.sh
adb -s 2ab58792 push source_me.sh /data/local/tmp/source_me_2.32.6.sh
