echo "Extracting DCT features for everything in $1."
path=$1
echo "Number of features: $2"
n=$2

$deltas=""
if [ $3 = "1" ]
    echo "Appending deltas"
    $deltas="-d"
fi

if [ $3 = "2" ]
    echo "Appending deltas + delta-deltas"
    $deltas="-dd"
fi

echo "Command: pyfile.py $path $n $deltas"