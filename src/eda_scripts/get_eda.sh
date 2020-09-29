# download eda.py
echo downloading eda.py
wget https://raw.githubusercontent.com/jasonwei20/eda_nlp/master/code/eda.py -O eda_orig.py

echo
echo verifying md5sum. If md5sum check fails, upstream resource has been updated. Please ensure eda.py corresponds to -- commit 1c5cda4 -- Feb 3, 2019
md5sum -c md5sums.txt

echo
patch eda_orig.py -i patch_file.patch -o eda.py

echo clearing directory
rm eda_orig.py
