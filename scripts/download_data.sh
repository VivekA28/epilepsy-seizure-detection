#!/bin/bash
SUBJECT=${1:-chb01}
mkdir -p data/raw

wget -r -N -c -np -nH --cut-dirs=3 \
  -P data/raw \
  https://physionet.org/files/chbmit/1.0.0/$SUBJECT/

echo "Done. Files in data/raw/$SUBJECT/"
