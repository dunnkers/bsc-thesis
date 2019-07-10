echo pwd # should be project dir

mkdir data_100x200
cd data_100x200
mkdir groundtruth
mkdir supervised
mkdir unsupervised
cd unsupervised
mkdir output
cd ../../

cd data/groundtruth
find . -maxdepth 1 -iname "*.png" | xargs -L1 -I{} convert -resize 100x200! "{}" ../../data_100x200/groundtruth/"{}"

cd ../supervised
find . -maxdepth 1 -iname "*.png" | xargs -L1 -I{} convert -resize 100x200! "{}" ../../data_100x200/supervised/"{}"

cd ../unsupervised/output
find . -maxdepth 1 -iname "*.png" | xargs -L1 -I{} convert -resize 100x200! "{}" ../../../data_100x200/unsupervised/output/"{}"