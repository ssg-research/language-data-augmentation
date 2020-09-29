wget https://raw.githubusercontent.com/kaushaltrivedi/fast-bert/master/sample_notebooks/new-toxic-multilabel.ipynb -O n\
otebook.ipynb

echo
echo verifying md5sum. If md5sum check fails, upstream resource has been updated. Please ensure eda.py corresponds to -\
- commit 1c5cda4 -- Feb 3, 2019
md5sum -c md5sums.txt

jupyter nbconvert --to script notebook.ipynb

diff notebook.py bert_helpers.py > patch_file.txt

echo
echo Number of lines in bert_helpers.py
wc -l bert_helpers.py

echo
echo Number of lines in jupyter script
wc -l notebook.py

echo
echo Number of line changes
wc -l patch_file.txt
