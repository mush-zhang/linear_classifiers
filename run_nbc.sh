#!/bin/bash
outFile1=./nbc_tmp.csv
outFile2=./avg_tmp.csv
outFile3=./base_tmp.csv
echo -n $'\n' > $outFile1
echo -n $'\n' > $outFile2
echo -n $'\n' > $outFile3
for perc in 1 10 50 70
do
  for i in {1..10}
  do
    python split.py yelp_cat.csv $perc
    OUTPUT="$(python nbc.py train.csv test.csv)"
    echo -n "${OUTPUT}" >> $outFile1
    echo -n ',' >> $outFile1
    OUTPUT="$(python avg.py train.csv test.csv)"
    echo -n "${OUTPUT}" >> $outFile2
    echo -n ',' >> $outFile2
    OUTPUT="$(python baseline.py train.csv test.csv)"
    echo -n "${OUTPUT}" >> $outFile3
    echo -n ',' >> $outFile3
  done
  echo -n $'\n' >> $outFile1
  echo -n $'\n' >> $outFile2
  echo -n $'\n' >> $outFile3
done
exit 0