#!/bin/sh

FOLDER=$1
FILES=`find $FOLDER -name "*.png"`

for file in $FILES; do
  if [ $(stat -c%s $file) -eq 216 ]; then
	# file is a 1x1 texture; convert it to 4x4 - etcpak can't handle non-divisible-by-4 textures
	convert $file -resize 4x4 $file
  fi
  ~/etcpak/build/etcpak -m -c bc7 -h dds $file ${file%.png}.dds &
done

# wait for all etcpak processes to finish
wait
