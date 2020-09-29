The file augmentation_wn.py uses WordNet synonym replacement for text data augmentation.
For inflecting Wordnet synonyms, augmentation_wn.py uses an inflection file (inflections.pkl)

The inflection file is created as follows, where "corpus" is a text corpus (either a single str or a list of sentences):
	save_inflection_dict(corpus, fpath=inflections.pkl) 

The function save_inflection_dict is found in the file make_inflections.py

--------------------------------------------------------------------------------

inflections.pkl has been produced with make_inflections.py from 5 million short English sentences (≤20 words), derived from the Stanford NLP parallel corpora, available at:
	https://nlp.stanford.edu/projects/nmt/

Stanford NLP parallel corpora are licensed with the Gnu General Public Licence.
