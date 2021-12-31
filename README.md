
# Credit risk assessment using South German credit data

Normally, most of the bank's wealth is obtained from 
providing credit loans so that a marketing bank must 
be able to reduce the risk of non-performing credit 
loans. The risk of providing loans can be minimized 
by studying patterns from existing lending data.
Hence creating a solution, which will utilize past 
experience to assist credit officers, can lead to 
less risk exposure of banks.\
The goal of this project is to create a ML solution 
for banking system of South German which assists 
credit officers in loan providing process (refuse 
or proceed) and calculates probability of default 
for an applicant.\
The final result of the project is a web application. 
It will take an applicant's characteristics as input 
and return default probability in line with a possible 
decision to make.

# Files' description
model.py contains model building code which actually 
generates all pkl files in the directory.\
test folder contains testing code.\
app.py contains web app code.\
templates folder contains web app template.\
SouthGermanCreditViz.ipynb contains all the visualizations and exploratory data analysis.\
woe_enc.py contains a function and a class to generate features in form of weight of evidence.\
documents folder contains all the documents related to the project.

## Deployment

The deployment is done using Heroku. Heroku is connected 
to the github. The web app will be updated automatically
if the code is updated on the github. The web app can be 
viewed by going [here](https://credit-risk-calculator.herokuapp.com/).


## Demo

![](https://github.com/VaheC/SouthGermanCredit/blob/main/gif_api_large.gif)

## Running Tests

Testing has been conducted only for a function and 
a class related to weight of evidence transformation.
The testing code is in test folder. To run the test go
to your command prompt/console and navigate to the test
folder. Then use the following command.

```bash
  pytest test.py
```


## Installation

If you have all packages from requirements.txt 
installed on your pc , then you can run all the codes
present in this project. 
    
## Run Locally
Install anaconda on your pc.\
Create a virtual environment in anaconda prompt. 

```bash
  conda -n env_name python=version

  example: conda -n sg_credit python=3.7

  python version can be taken from requirements.txt
```

Activate the virtual environment in anaconda prompt

```bash
  activate env_name 

  example: activate sg_credit 
```

Clone the project in command prompt

```bash
  git clone https://link-to-project
```

Go to the cloned project directory on your pc 
in anaconda prompt

```bash
  cd directory-of-project

  example: cd C:\Users\Desktop\SouthGermanCredit
```

Install dependencies in anaconda prompt

```bash
  pip install -r /path/to/requirements.txt
```

Run the model building code in anaconda prompt

```bash
  python model.py
```

Start the web app on your pc from anaconda prompt

```bash
  python app.py
```
Use http shown in anaconda prompt (the last line of 
output after running the code above) to open the web
app in a web browser.

```bash
  example: http://127.0.0.1:5000/
```
## Acknowledgements

- [How to write a Good readme](https://readme.so)

