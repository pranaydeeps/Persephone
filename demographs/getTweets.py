# This tests the classifier without front end
import sys
from classify import classified
import nltk
import sys
handle = str(sys.argv[1])
py_obj = classified(handle)
print(py_obj)

