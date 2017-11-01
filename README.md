# BBC Classification
## Classification of BBC stories into category using NLTK

The data come from http://mlg.ucd.ie/datasets/bbc.html (BBC dataset, raw) and
have been moved into directory `news`.

`199.txt` in `sport` had character <A3> (the pound symbol) that needed
converting, with the following: `iconv -f ISO-8859-1 -t UTF-8 199.txt`.

Run `learn.py` with `PYTHONIOENCODING=utf-8` and `NLTK_DATA` and `PYTHONPATH`
so that the nltk libraries and featx from the NLTK3 Cookbook code can be used.

Also ensure availability of `megam` in PATH if using maxent; naive Bayes is on
by default.
