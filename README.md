# Email Sherlock:
### Using Machine Learning to Extract Information from Large Email Datasets.

Email Sherlock is specialized search engine that help investigation involving large emails dataset. It works with two steps:  First, it separetes converstaions from noise by clustering with DBSCAN, then it builds a query expation with word2vec.

We believe Email Sherlock can adapted to prevent frauds and avoid classified information being spread.


### Summary
This repo comtains the necessary code to reproduce the results and run the WebApp on your localhost.
The code is organazed in tree folders:

- [email-download](): Downloads pdfs, converts them to text and store the data in SQLite.
- [model-script](): Python scripts to clean the data and build the models.
- [web-app](): The code necessary to run the WebApp.


### Presentation:
The slides are available at [email-sherlock-slides](http://www.slideshare.net/gondinjose/email-sherlock)
and can also what the video at [email-sherlock-video](https://youtu.be/aibXSdmqaY8).



