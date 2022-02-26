# Solve Sudoku from an Image
This project aims to take any picture of a sudoku puzzle, extract its contents using image processing techniques, solve the puzzle, and then overlay the solution on the original image.

![Selected corners, solved puzzle](https://github.com/jpritcha3-14/sudoku-image-solver/blob/master/result.png)

## Software Used
- anaconda / conda - Used to set up isolated environments with specific packages. Can be [downloaded here](https://www.anaconda.com/distribution/). 
- scikit Learn - Included in the base conda environment.
- openCV - Can be installed from within the conda environment using `pip install opencv-python`

## How To Use
Once the environment is set up correctly, you can run the program on a sudoku image using the default classifier by running:

`python3 sudokuImageSolver.py -i <path to image>`

Select the 4 corners of the sudoku puzzle in the pop up and press enter.  Then press q to quit the program.

To train a classifier on your own sudoku images, read the [HOWTO.md file](https://github.com/jpritcha3-14/sudoku-image-solver/blob/master/modelTraining/HOWTO.md)
in the modelTraining directoty.  

## Goals and Progress
- **Parse individual squares from a Sudoku puzzle, accounting for skew in camera angle and warping in the paper.**
	- Complete!  User selects the 4 corners of the puzzle with mouse clicks. These points are then used to transform the puzzle to a square of a standard size.  Matching against a template square is used to find the location of all squares while accounting for slight curves in the paper.

- **Determine the numbers contained in parsed squares.**
	- ~~Currently using template matching against a single set of digit images to classify numbers.  While it is a relatively effective strategy for this constrained of a problem, it has issues with similarly shaped 6s, 9s and 8s.~~
    - Classifier (random forest) has been trained on ~900 numbers parsed from images gathered from the internet.  Cross validated accuracy for digit recogintion is around 98%.  Will continue to gather images to improve classifier.

- **Solve the Sudoku puzzle**
	- ~~The solvePuzzle.py script solves a Sudoku puzzle represented as numpy ndarray using a brute force backtracking method.  This method is not viable for less constrained puzzles and takes too long.  Logic needs improvemnet to solve more difficult puzzles.~~
    - Complete! Adapted code from [Peter Norvig's excellent post on the problem](https://norvig.com/sudoku.html) to rewrite the Sudoku solver.  The new algorithm uses a recursive search in conjunction with constraint propogation to limit the scope of the solution search.

- **Overlay the solution to the puzzle on the original image**
	- Complete!

## Next Steps
- ~~Create a script to ingest Sudoku images and create examples of each number using current functions for corner selection and square parsing.~~
- ~~Find, ingest, and classify digit images to train a learning model.~~
- ~~Train a supervised learning model to correctly identify numbers.~~
- ~~Use the trained model to classify numbers from new puzzles.~~
- ~~Overlay the results of solving the sudoku puzzle on the original image.~~
- ~~Improve puzzle solver logic to work with less constrained (harder) puzzles~~
- Clean up and comment code, run a linter. 
- Continue to gather and ingest sudoku images to improve the classifier.
