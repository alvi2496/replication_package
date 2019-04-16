# Replication of Do They Discuss Design?
This study aims to replicate the findings in [1](#references).

## Environment Setup

### Requirements
1. Python 3.6.0
2. Virtual Environment

### Setup Virtual Environment 
To install the virtual environment run   
`$ python3 -m venv venv`  
To activate this virtual environment run  
`$ source venv/bin/activate`

### Install Requirements
The requirements are defined in requirements.txt with version. To 
install them run  
`$ pip install -r requirements.txt`

### Download `nltk stopwords corpus`
To download the `nltk stopwords corpus` run  
`nltk.download('stopwords')` in the `python interactive console`.


## Run the Replication
1. To run the classifier: `python classification.py`
2. Research Question 1: `python rq_1.py`
3. Research Question 2(a): `python rq_2_a.py`
4. Research Question 2(b): `python rq_2_b.py`
5. Research Question 2(c): `python rq_2_c.py`


## About Paper
The paper can be found in this folder names as `paper.pdf`


## About Data
The data file can be found in `data` folder.


## Results
The outputs will be generated in the results folder. Please refer to the
`output` folder for all the output files. Please click [here](https://docs.google.com/spreadsheets/d/1YfRCcBOrMyNlYLrEMaG6Dj8r9hUqJoaVtatcKi34JxM/edit?usp=sharing) to get the analytical results.


## Report
The `report.pdf` contains the report of the replication in pdf format. The latex file package can be found
in the `report` forlder.


## References
1. J. a. Brunet, G. C. Murphy, R. Terra, J. Figueiredo, and D. Serey,
“Do developers discuss design?” in Proceedings of the 11th Working
Conference on Mining Software Repositories, ser. MSR 2014. New
York, NY, USA: ACM, 2014, pp. 340–343.
