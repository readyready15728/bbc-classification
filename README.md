# BBC Classification
## Classification of BBC stories into category using NLTK

The data come from http://mlg.ucd.ie/datasets/bbc.html (BBC dataset, raw) and
have been moved into directory `news`.

`199.txt` in `sport` had character `<A3>` (the pound symbol) that needed
converting, with the following: `iconv -f ISO-8859-1 -t UTF-8 199.txt`.

The library featx from the
[repository](https://github.com/japerk/nltk3-cookbook/) for *NLTK3 Cookbook* is
required for this code to function. Install that first and then run `learn.py`
with and `NLTK_DATA` and `PYTHONPATH` set up so that the nltk libraries and
featx can both be used.  A sample invocation might look like this:

```bash
PYTHONPATH=~/src/nltk3-cookbook/ NLTK_DATA=~/.nltk_data python3 learn.py
```

(Directions to use `NLTK_DATA` can be ignored if the data are in a default
location like `$HOME/nltk_data` but ew.)

The model will be pickled and saved as `fit`, making it available for usage on
the next invocation; delete `fit` to create a new model.

Also ensure availability of `megam` in `PATH` if using maxent; naive Bayes is on
by default.
