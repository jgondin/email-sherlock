#!/usr/bin/env python
# encoding: utf-8

"""
pdfTextToDatabase.py

1. Get a list of PDF names from the database
2. Open them
3. Extract text contents
4. Write contents back to the database

"""

from hrcemail_common import *
import string
import slate
import sys
from subprocess import call

def extract(filename):
	try:
		#file = open('pdfs/'+filename+'.pdf', 'rb')
		call(['pdftotext', 'pdfs/' + filename + '.pdf', 'txts/' + filename + '.txt'])
	except:
		return None;

	try:
		#this removes non-ASCII characters. Might not be desirable in some circumstances.
		#return filter(lambda x: x in string.printable,"\n".join(slate.PDF(file)))
		return open('txts/' + filename + '.txt', 'rb').read()
	except:
		return None;
		
docIDs = Document.select(Document.docID).where(Document.docText >> None)

for docID in docIDs:
	filename = docID.docID
	
	print "Working on", filename
	output_string = extract(filename)
	insert_query = Document.update(docText = output_string).where(Document.docID == filename)
	insert_query.execute()
