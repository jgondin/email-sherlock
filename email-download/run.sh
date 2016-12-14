#!/bin/bash
mkdir -p ../../hrc_emails/pdfs/
mkdir -p ../../hrc_emails/txts/
mkdir -p ../../hrc_emails/zips/
source virt-hrcemail/bin/activate

python email-download/downloadMetadata.py
python email-download/generatePDFList.py
if [ "$1" = "no-pdf-download" ] 
then 
    echo "skipping PDF download"
else
    cd pdfs/
	wget --no-check-certificate --no-clobber --timeout=5 --tries=20 -i ../pdflist.txt
	cd ..
fi
python email-download/zipPDFs.py
python email-download/pdfTextToDatabase.py
deactivate
