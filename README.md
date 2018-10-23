# EE542-lab10
lab10 machine learning on genomics data

This repository contains python codes for checking integrity, parsing data, generating meta data, generating miRNA matrix and performing ML on the given dataset.

Once the data is downloaded using the command .<path-to-gdc-client>/gdc-client download –m <path-to-manifest-file> from the genomics repository. 
1) Run the check.py file. The file needs to execute where the dataset is present
All files are compatible with Python 3 and above versions
eg : $python3 check.py

2) Run the parse_file_case_id.py script to parse the data. 
Next get the JSON file from the genomimcs data repository. 

3) Run the request_meta.py to get the meta data

4) Next Run the gen_miRNA_matrix.py to get the miRNA matrix

5) Once the miRNA matrix is obtained, Run the test1.py file for running the machine learning algorithm. 
We have performed KNN clustering. 
The precision , accuracy , F1-score and sensitivity values can be seen in the graph. 

