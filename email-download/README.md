# Get and analyze Hillary Clinton's email

In response to a public records request, the U.S. State Department is releasing Hillary Clinton's email messages from her time as secretary of state. Every month, newly released messages are posted to [foia.state.gov](https://foia.state.gov/) as PDFs, with some metadata.

## What's in the toolkit
* **run.sh** runs all of the Python scripts in the toolkit automatically, allowing easy updates when messages are released.

* **downloadMetadata.py** scrapes sender, recipient, message date and subject from [the message list](https://foia.state.gov/Search/Results.aspx?collection=Clinton_Email) and writes this metadata to a sqlite database, `hrcemail.sqlite`.
* **generatePDFList.py** writes `pdflist.txt`, a newline-delimited list of HTTPS URLs of the message PDFs.
* **zipPDFs.py** makes a zip file of PDFs for each release of messages.
* **pdfTextToDatabase.py** extracts text from the PDF files (which are OCR'd by State) and writes the text to a sqlite database, `hrcemail.sqlite`.

* **HRCEMAIL_names.csv** is a list that pairs sender and recipient names provided by the State Department website with that person's commonly-used name. For example, `HRC` becomes `Hillary Clinton`.


Install [virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/) if necessary.
```
pip install virtualenv
```

Create a virtual environment. **Python 2.7.9** is required, specifically for SSL (HTTPS) support. State Department's website requires HTTPS.
```
virtualenv -p /usr/bin/python2.7 virt-hrcemail
source virt-hrcemail/bin/activate
```

Install all the Python dependencies. 
```
pip install -r email-download/requirements.txt
```

Install pdftotext.
```
<<<<<<< HEAD
sudo apt-get update
sudo apt-get install

```

Then, run the shell script.
=======
./email-download/run.sh
```

You will need `wget` to download the PDFs and `pdftotext` to convert pdfs to texts.
>>>>>>> b26227904c86eea74a0255fef9adafdf1f848587

```
./run.sh
```
<<<<<<< HEAD

You will need `wget` to download the PDFs. (Mac OS X users can install it using [homebrew](http://brew.sh/).)

The pdf and txt files will downloaded  in ```../data/pdfs``` and ```.../data/txts```, respectively. Make sure you have permition and space available in you computer.


Finally, load `HRCEMAIL_names.csv` into the `hrcemail.sqlite` database.
```
csvsql --db "sqlite:///hrcemail.sqlite" --insert --no-create --blanks --table name  HRCEMAIL_names.csv 
```


This based on ```https://github.com/wsjdata/clinton-email-cruncher.git```, which was slitely modified.
=======
>>>>>>> b26227904c86eea74a0255fef9adafdf1f848587
