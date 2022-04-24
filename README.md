# PIC16B-Project

## Project proposal

### Abstract
We plan to perform sentiment analysis on the written opinions of Supreme Court Justices, with the aim to differentiate and highlight the unique "legal writing styles" of the Justices, which will be beneficial for people learning about legal writing and may reveal Justices' legal philosophy. Our methodology will include downloading large volumes of Supreme Court opinion PDFs from the official website. Then, we would use OCR tools to detect and store the text before using regular expressions to separate the opinions and identify the author in order to construct our official dataset CSV. After preparing the data, we would utilize an NLP package in order to find high prevalence words for each author, as well as score the overall sentiment in the opinion.
GitHub Repository: https://github.com/RaymondBai/PIC16B-Project 
### Planned Deliverables
The intended deliverable is a report on the legal writing styles of the 12 Justices who served on the Roberts court between 2005 and 2014.  The deliverable may take on the form of an academic report or a more reader-friendly article; the purpose, regardless of format, is to provide an introduction to legal writings for law students and the general public interested in appellate litigation. Note that this project does not directly attempt to classify or predict the Justices’ judicial philosophies, but the result from textual and sentiment analysis may inadvertently provide insight into exactly that.
Full Success: Download Justices’ opinions and parse/clean them into pure text files; calculate Flesch-Kincaid Readability Score (Flesch-Kincaid), compile individual Justice’s keyword frequency, and calculate overall sentiment.
Partial Success: We anticipate the PDF download, Optical character recognition (OCR) to plain text, and proper text parsing/cleaning will pose the greatest challenges. Thus at the minimum, we can create a function that allows the user to search for and download specific Supreme appellate opinion PDFs before reading and parsing the file into a plain text file for further analysis.
### Resources Required
In the archive of Supreme Court opinions (SCOTUS Archive), the official opinions are separated by terms in large PDFs. There is no special resource required in terms of cloud service or accounts with subscriptions. We may need computing power greater than that of our laptops in order to run the function that automatically downloads large files and performs text parsing/cleaning.
### Tools and Skills Required
We certainly need PIC 16A Python skills (Pandas, NumPy),  libraries for downloading files from websites (Request), OCR tools for reading text from PDF files (such as Keras-OCR, Pytesseract), and natural language processing tools (such as NLTK, SpaCy). If necessary, Matplotlib and Seaborn will be used for visualizations. 
### What You Will Learn
For Raymond: I certainly hope to become a lot more adept at writing functions that can automate tasks (in this case downloading files and compiling into data frames); I also will gain valuable experience in natural language processing.
For Ashwin: As NLP was a very small portion of PIC 16A, I hope to gain a more thorough understanding through an in-depth analysis of a real-world dataset using NLP packages. Additionally, being able to take a project from initialization to completion will help me gain a better understanding of data analysis procedures.
For James: I would like to learn to employ machine learning with a wide variety of data structures. This project allows me to work with a new input, expanding my knowledge of Natural Language Processing and most generally, expanding my ability to use programming to analyze human behavior. 
For Jillian: I would like to become better at implementing NLP. I also want to learn how to perform a sentimental analysis because I think this would be a great thing to know when I enter the industry. I am also interested in human behavior and communication which is why I want to learn how to conduct sentiment analysis. I also want to be stronger in machine learning.
### Risks
Due to the format of the PDF, there may be significant difficulty in data cleaning and parsing after OCR tools capture the text, leading to the inability to separate opinions, discern the opinion author, etc (just to name a few possibilities) from regular expressions.
At the moment, lack of experience with performing a sentiment analysis could be an issue. However, once we learn more about NLP we believe that everything has the chance to go smoothly once we complete the challenging amount of data cleaning and preparation.
### Ethics
We do not anticipate a specific group of people that would be harmed by the project. However, as we mentioned earlier, our results should not be construed as a concrete analysis of the Justices' political philosophy or stance on social issues, since the misinterpretation is likely to lead to non-factually supported discussions on the qualification of justices. If misinterpreted, this could provide some ethical issues as it could lead to unjust judgments of Justices that are not necessarily true.
A common risk of sentiment analysis is the inability to contextualize tone and sarcasm. While encountering sarcasm is highly unlikely in the texts we are using, incorrectly identified sentiments could mislead users. It may be important to consider the target of statements. For example, there’s a difference between “he prevented a horrible incident” or “his actions were horrible. ” If the system is only trained on adjectives, it could identify both of these statements to be negative, when one of these statements is positive. This could extend to identifying a neutral opinion as positive or negative, as well. It is important that our system is precise enough to consider and distinguish contexts. If our product decodes text incorrectly, this would deceive users, potentially incorrectly sway the options of the writers of the reports, or lead to misinterpretation of the cases and people described in the texts. 

### Potentially useful links
https://fivethirtyeight.com/features/which-justices-were-bffs-this-scotus-term/
https://basilchackomathew.medium.com/best-ocr-tools-in-python-4f16a9b6b116
https://mqscores.lsa.umich.edu/replication.php
http://supremecourtdatabase.org/index.php

