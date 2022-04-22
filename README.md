# PIC16B-Project

## Project proposal

### Abstract
We plan to perform a sentiment analysis on the written opinions of Supreme Court Justices, with the aim to differentiate and highlight the unique "legal writing styles" of the Justices, which will be beneficial for people learning about legal writing and may reveal Justices' legal philosophy. Our methodology will include downloading large volumes of Supreme Court opinion PDFs from the official website. Then, we would use OCR tools to detect and store the text before using regular expressions to separate the opinions and identify the author in order to construct our official dataset CSV. After preparing the data, we would utilize an NLP package in order to find high prevalence words for each author, as well as score the overall sentiment in the opinion.
GitHub Repository: https://github.com/RaymondBai/PIC16B-Project 
### Planned Deliverables
The intended deliverable is a report on the legal writing styles of the 12 Justices who served on the Roberts court between 2005 and 2014.  The deliverable may take on the form of an academic report or a more reader-friendly article; the purpose, regardless of format, is to provide an introduction to legal writings for law students and the general public interested in appellate litigation. Note that this project does not attempt to classify or predict the Justices’ judicial philosophies, but the result from textual and sentiment analysis may inadvertently provide insight into exactly that.
Full Success: Download Justices’ opinions and parse/clean into pure text files; calculate Flesch-Kincaid Readability Score (Flesch-Kincaid), compile individual Justice’s keyword frequency, and calculate overall sentiment.
Partial Success: I anticipate the PDF download, Optical character recognition (OCR) to plain text, and proper text parsing/cleaning will pose the greatest challenges. Thus at the minimum, we can create a function that allows the user to search for and download specific Supreme Court opinion PDFs before reading and parsing the file into a plain text file for further analysis.
### Resources Required
In the archive of Supreme Court opinions (SCOTUS Archive), the official opinions are separated by terms in large PDFs. There is no special resource required in terms of cloud service or accounts with subscriptions. We may need computing power greater than that of our laptops in order to run the function that automatically downloads large files and performs text parsing/cleaning.
### Tools and Skills Required
We certainly need PIC 16A Python skills (Pandas, NumPy),  libraries for downloading files from websites (Request), OCR tools for reading text from PDF files (such as Keras-OCR, Pytesseract), and natural language processing tools (such as NLTK, SpaCy). If necessary, Matplotlib and Seaborn will be used for visualizations. 
### What You Will Learn
For Raymond: I certainly hope to become a lot more adept at writing functions that can automate tasks (in this case downloading files and compiling into data frames); I also will gain valuable experience in natural language processing.
For Ashwin: INSERT HERE
For James: INSERT HERE
### Risks
Due to the format of the PDF, there may be significant difficulty in data cleaning and parsing after OCR tools capture the text, leading to inability to separate opinions, discern the opinion author, etc (just to name a few possibilities) from regular expressions
INSERT HERE
### Ethics
INSERT HERE
All projects we undertake involve decisions about whose interests matter; which problems are important; and which tradeoffs are considered acceptable. Take some time to reflect on the potential impacts of your product on its users and the broader world. If you can see potential biases or harms from your work, describe some of the ways in which you will work to mitigate them. Remember that even relatively simple ideas can have unexpected and impactful biases.

### Other useful links
https://fivethirtyeight.com/features/which-justices-were-bffs-this-scotus-term/
https://basilchackomathew.medium.com/best-ocr-tools-in-python-4f16a9b6b116
https://mqscores.lsa.umich.edu/replication.php
http://supremecourtdatabase.org/index.php
