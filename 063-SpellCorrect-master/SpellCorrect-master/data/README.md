# Notes About the Data

## big.txt

This is the original data provided by Peter Norvig.  It remains unmodified.

## en_ANC.txt.bz2

This is every plain-text file from the [Open American National Corpus](http://www.americannationalcorpus.org/oanc/index.html) version 1.0.1, concatenated together, with extraneous newlines removed and some non-ASCII space characters replaced with ASCII ones.  Otherwise it is unmodified and is UTF-8.  Contains approximately 11,484,017 words of written, native, American English.

## Testing and Development Data

For testing, we use two data sets: a portion of the [Birkbeck spelling error corpus](http://ota.ox.ac.uk/headers/0643.xml) and the [Aspell testing data](http://aspell.net/test/common-all/batch0.tab).  For development, only the Birkbeck corpus is used.

Here is how the testing and development subsets from the Birkbeck corpus were generated.  For starters, not every sub-corpus is used here; the following are **excluded**:

* ASHFORD
* CHES
* GATES
* HOLBROOK
* MASTERS
* NFER1
* PERIN1
* PETERS1
* PETERS2
* SAMPLE (since it is just the first 10 lines of every sub-corpus)

This resulted in 12,533 unique word-error pairs, which was then filtered for **wrong-word** errors, as in, the error is actually a correctly-spelled word, but it is not the right word for this context.  While this is an interesting and important topic, it is beyond the scope of this project, especially since many wrong-word errors require a great deal of external knowledge.  For example, according to this corpus, at least, one would need to know that in British English, one would say "a heavy _woolen_ coat", while in American English, "a heavy _wool_ coat".  A word-error pair was determined to be a wrong-word-error pair if the "error" was not flagged as an error by ``pyenchant`` (yeah, yeah), using the provided `en_US` and `en_GB` dictionaries.  3,488 word-error pairs were identified, or 27.8% of this subset of the Birkbeck corpus.  These can be found in `Birkbeck_subset_wrong_word_errors.csv`.

We were also interested in testing the claim that "80 to 95% of spelling errors are an edit distance of 1 from the target", and also Norvig's claim that, within his development set of 270 errors, this was only true for 76% of them.  Over this entire set of 9,045 word-error pairs, we computed the edit distance between each pair using [nltk](http://www.nltk.org/api/nltk.metrics.html#nltk.metrics.distance.edit_distance), the results of which are provided along with the testing and development sets here.  Here's a summary of what we learned:

* mean edit distance: 1.74
* median edit distance: 1
* % of spelling errors that are an edit distance of 1 away from the target: 60.98%

We then generated a random 60/40 split into development/testing, with `$ tail -n +2 Birkbeck_subset_spelling_errors_all.csv | shuf | split -l 5427`.  They can be found in `Birkbeck_subset_spelling_errors_development_set.csv` and `Birkbeck_subset_spelling_errors_testing_set.csv`, respectively.  Only 40% of the Birkbeck corpus is reserved for testing because Aspell's testing data provides an additional 4,206 test cases, for a total of 7,824 test cases and 5,427 development cases.
