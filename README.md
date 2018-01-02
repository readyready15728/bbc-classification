# BBC Classification
## Classification of BBC stories into category using NLTK

The data come from http://mlg.ucd.ie/datasets/bbc.html (BBC dataset, raw) and
have been moved into directory `news`.

`199.txt` in `sport` had character <A3> (the pound symbol) that needed
converting, with the following: `iconv -f ISO-8859-1 -t UTF-8 199.txt`.

Run `learn.py` with `PYTHONIOENCODING=utf-8` and `NLTK_DATA` and `PYTHONPATH`
set up so that the nltk libraries and featx from the NLTK3 Cookbook code can be
used.  A sample invocation might look like this:

```bash
PYTHONIOENCODING=utf-8 PYTHONPATH=~/src/nltk3-cookbook/ NLTK_DATA=~/.nltk/ python learn.py
```

The model will be pickled and saved as `fit`, making it available for usage on
the next invocation.

Also ensure availability of `megam` in `PATH` if using maxent; naive Bayes is on
by default.
