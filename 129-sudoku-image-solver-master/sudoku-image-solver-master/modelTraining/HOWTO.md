# How to train the model
## 1. Ingest 

The `ingestImages.py` script parses individual numbers out of Sudoku images and places them in an output directory.  Run the command as follows:

`python3 ingestImages.py <output dir> <files...>`

For each file specified, the Sudoku image will appear in a window.  Manually select the 4 corners of the puzzle with the cursor and press enter.  The script prints out a list of files written to the output directory and appends '_ingested' to the input image file.
## 2. Sort

The `manualNumberSort.py` script helps with sorting the ingested images from the Sudoku puzzle into files.  Run the command as follows:

`python3 manualNumberSort.py <output dir> <files...>`

The script will create folders 10 subdirectories (0 to 9) in the output directory if they do not already exist.  The script will then ask you to classify numbers by pressing the correct key on the keyboard.  The image file is then moved to that subdirectory.
## 3. Train

The `trainModel.py` script trains a random forest classifier on the sorted images and then saves the classifier as a joblib file.  Run the command as follows:

`python3 trainModel.py -i <path to images> -n [<output name>]`

The path to image argument should be the same directory used for the output of the manualNumberSort.py script (and should have 10 subdirectories labelled 0 to 9).  If the output name is not specified it will default to classifier.joblib.
