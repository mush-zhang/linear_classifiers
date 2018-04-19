#!/bin/bash
outFile=./nbc_tmp.csv
echo -n $'\n' > $outFile
for perc in 1 10 50 70
do
  for i in {1..10}
  do
    python split.py yelp_cat.csv $perc
    OUTPUT="$(python nbc.py train.csv test.csv)"
    echo -n "${OUTPUT}" >> $outFile
    echo -n ',' >> $outFile
  done
  echo -n $'\n' >> $outFile
done
exit 0