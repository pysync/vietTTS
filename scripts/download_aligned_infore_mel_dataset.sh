data_root="./train_data" # modify this
pushd .
mkdir -p $data_root
cd $data_root
gdown --id 1bofud5YC4WbzaHjlbHd4kQJJ4d6FRhOZ -O infore.zip
unzip infore.zip 
popd