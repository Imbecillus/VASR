# This script extracts DCT features.
# Usage: extract_dct_all.sh [path to TCD-TIMIT volunteer set] [number of features] [[0: no deltas; 1: deltas; 2 delta-deltas]]

echo "Extracting DCT features for everything in $1."
path=$1
echo "Number of features: $2"
n=$2

deltas=""
if [ $3 = "1" ]
then
    echo "Appending deltas"
    deltas="-d"
fi

if [ $3 = "2" ]
then
    echo "Appending deltas + delta-deltas"
    deltas="-dd"
fi

echo "Command: python extract_dct_features.py $path/[SPEAKER]/Clips/ROI/[SEQUENCE]/ $n $deltas"

for d in $path*M/ $path*F/
do
    for seq in ${d}Clips/ROI/*/
    do
        echo "$seq"
        python extract_dct_features.py $seq $n $deltas
    done
done
