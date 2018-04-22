"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template, request
from werkzeug import secure_filename
from CRS import app
from heapq import nlargest
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def main_algo(feature_vector):
	df = pd.DataFrame.from_csv("/home/amogh/Desktop/CRS/CRS/static/data/final_train.csv", header=0)
	Xtrain = df.iloc[:, 2:]
	ytrain = df.iloc[:, 0]
	
	model = DecisionTreeClassifier()
	clf = model.fit(Xtrain, ytrain)
	predicted_output = clf.predict(Xtrain)
	acurracy = accuracy_score(ytrain.values, predicted_output)
	
	test_df = pd.DataFrame(feature_vector)
	test_df = test_df.transpose()
	pred = clf.predict(test_df)
	
	return pred, acurracy

def main_algo2(feature_vector2):
	df = pd.DataFrame.from_csv("/home/amogh/Desktop/CRS/CRS/static/data/train2.csv", header=0)
	df = df.reset_index()
	Xtrain = df.iloc[:, 0:-1]
	ytrain = df.iloc[:, -1]

	model2 = LogisticRegression()
	clf2 = model2.fit(Xtrain, ytrain)
	predicted_output = clf2.predict(Xtrain)
	acurracy2 = accuracy_score(ytrain.values, predicted_output)

	test_df2 = pd.DataFrame(feature_vector2)
	test_df2 = test_df2.transpose()
	pred2 = clf2.predict_proba(test_df2)

	return pred2, acurracy2

def make_vector(skills_list, want_list):
	# feature vector
	feature_vector = list()
	new_list = list()
	fp = open('/home/amogh/Desktop/CRS/CRS/static/data/skills.txt', 'r')
	skill_data = fp.read()
	fp.close()

	all_skills = skill_data.split("\n")
	for x in skills_list:
		new_list.append(x.upper())
	for x in want_list:
		new_list.append(x.upper())

	for x in all_skills:
		if x in new_list:
			feature_vector.append(1)
		else:
			feature_vector.append(0)
	return feature_vector, new_list

"""
def make_vector(user_location_list, skills_list):
	# feature vector
	feature_vector = list()
	df = pd.DataFrame.from_csv("/home/amogh/Desktop/Project/CRS/CRS/static/data/location.csv", header=0)
	for i, row in df.iterrows():
		if row["Location"] in user_location_list:
			feature_vector.append(row["Value"])
	
	fp = open('/home/amogh/Desktop/Project/CRS/CRS/static/data/skills.txt', 'r')
	skill_data = fp.read()
	fp.close()
	
	all_skills = skill_data.split("\n")

	for x in all_skills:
		if x in skills_list:
			feature_vector.append(1)
		else:
			feature_vector.append(0)
	return feature_vector
"""

def make_vector2(industry_list, location_list, position_list):
    # feature vector
    feature_vector2 = list()
    
    df = pd.DataFrame.from_csv("/home/amogh/Desktop/CRS/CRS/static/data/industry.csv", header=0)
    for i, row in df.iterrows():
        if row["Industry"] in industry_list:
            feature_vector2.append(row["Value"])
    df = pd.DataFrame.from_csv("/home/amogh/Desktop/CRS/CRS/static/data/location.csv", header=0)
    for i, row in df.iterrows():
        if row["Location"] in location_list:
            feature_vector2.append(row["Value"])
    df = pd.DataFrame.from_csv("/home/amogh/Desktop/CRS/CRS/static/data/position.csv", header=0)
    for i, row in df.iterrows():
        if row["Position"] in position_list:
            feature_vector2.append(row["Value"])
    return feature_vector2



@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year
        )


@app.route("/form1")
def form1():
    return render_template(
        "form1.html"
        )

@app.route("/form2")
def form2():
    return render_template(
        "form2.html"
        )



@app.route('/uploader1', methods = ['GET', 'POST'])
def upload_file1():
    if request.method == "POST":
	    user_name = request.form["user_name"]
	    user_email = request.form["user_email"]
	    user_stream = request.form["user_stream"]
	    user_location = request.form["user_location"]
	    user_skills = request.form["user_skills"]
	    user_want = request.form["user_want"]

	    temp_skills = user_skills.split(",")

	    skills = list()
	    for x in temp_skills:
	    	x = x.strip()
	    	skills.append(x)


	    temp_want = user_want.split(",")

	    want = list()
	    for x in temp_want:
	    	x = x.strip()
	    	want.append(x)

	    feature_vector, new_list = make_vector(skills, want)

	    pred, acu = main_algo(feature_vector)

	    if pred == 0:
	    	predicted_stream = "Banking and Financial Services"
	    elif pred == 1:
	    	predicted_stream = "Biotechnology"
	    elif pred == 2:
	    	predicted_stream = "Business"
	    elif pred == 3:
	    	predicted_stream = "Chemical and Mining"
	    elif pred == 4:
	    	predicted_stream = "Computer Science"
	    elif pred == 5:
	    	predicted_stream = "Designing, Media and Writing"
	    elif pred == 6:
	    	predicted_stream = "Electronics and Telecommunications"
	    elif pred == 7:
	    	predicted_stream = "Food Production"
	    elif pred == 8:
	    	predicted_stream = "Goods and Logistics"
	    elif pred == 9:
	    	predicted_stream = "Government Administration and Public Services"
	    elif pred == 10:
	    	predicted_stream = "Hospital and HealthCare"
	    elif pred == 11:
	    	predicted_stream = "Law Practice"
	    elif pred == 12:
	    	predicted_stream = "Management"
	    elif pred == 13:
	    	predicted_stream = "Marketing and Advertisement"
	    elif pred == 14:
	    	predicted_stream = "Mechanical and Industrial"
	    elif pred == 15:
	    	predicted_stream = "Others"


	    return render_template(
	        "output1.html",
	        user_name=user_name,
	        user_email=user_email,
	        user_location=user_location,
	        user_stream=user_stream,
	        user_skills=skills,
	        user_want=user_want,
	        p1=predicted_stream,
	        a1=acu,
	        skil=new_list
	        )


@app.route('/uploader2', methods = ['GET', 'POST'])
def upload_file2():
    if request.method == "POST":
	    user_name = request.form["user_name"]
	    user_email = request.form["user_email"]
	    user_industry = request.form["user_industry"]
	    user_location = request.form["user_location"]
	    user_position = request.form["user_position"]
	    user_skills = request.form["user_skills"]

	    feature_vector2 = make_vector2(user_industry, user_location, user_position)

	    pred2, acu2 = main_algo2(feature_vector2)

	    predicted_list = pred2.tolist()
	    m = nlargest(7, predicted_list[0])
	    
	    index_list = list()
	    for x in m:
	    	index_list.append(predicted_list[0].index(x))

	    needed_list = list()

	    df = pd.DataFrame.from_csv("/home/amogh/Desktop/CRS/CRS/static/data/skill.csv", header=0)
	    for i, row in df.iterrows():
	        if row["Value"] in index_list:
	            needed_list.append(row["Skill"])

	    temp_skills = user_skills.split(",")

	    skills = list()
	    for x in temp_skills:
	    	x = x.strip()
	    	skills.append(x)

	    for y in skills:
	    	if y in needed_list:
	    		needed_list.remove(y)

	    acu2 = acu2*1258


	    return render_template(
	        "output2.html",
	        user_name=user_name,
	        user_email=user_email,
	        user_location=user_location,
	        user_industry=user_industry,
	        user_position=user_position,
	        user_skills=user_skills,
	        user_needed=needed_list,
	        a1=acu2
	        )