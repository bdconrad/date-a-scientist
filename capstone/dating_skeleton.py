import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC



### Dataset Exploration

#Creating dataframe:
df = pd.read_csv("profiles.csv")

plt.hist(df.income, bins=25)
plt.title('Income of Users')
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.show()

plt.hist(df.age, bins=20)
plt.title('Age of Users')
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.xlim(16, 80)
plt.show()

plt.scatter(df.age, df.income, alpha=0.5)
plt.title('Age vs Income')
plt.xlabel("Age")
plt.ylabel("Income")
plt.show()



### Creation of New Columns 

# create numerical data for drinking habits
drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df["drinks_code"] = df.drinks.map(drink_mapping)

# create numerical data for smoking habits
smoke_mapping = {"no": 0, "trying to quit": 1, "sometimes": 2, "when drinking": 3, "yes": 4}
df["smokes_code"] = df.smokes.map(smoke_mapping)

# create numerical data for drug habits
drug_mapping = {"never": 0, "sometimes": 1, "often": 2}
df["drugs_code"] = df.drugs.map(drug_mapping)

# creates numerical data describing whether or not an individual has at least one child
	# 1 = has one or more children
	# 0 = has no children
has_kids_mapping = {
	"has kids, and wants more": 1, "has a kid, and wants more": 1, "doesn&rsquo;t have kids, but wants them": 0,
	"has kids, and might want more": 1, "has a kid, and might want more": 1, "doesn&rsquo;t have kids, but might want them": 0,
	"has kids, but doesn&rsquo;t want more": 1, "has a kid, but doesn&rsquo;t want more": 1, "doesn&rsquo;t have kids, and doesn&rsquo;t want any": 0,
	"has kids": 1, "has a kid": 1, "doesn&rsquo;t have kids": 0,
	"wants kids": 0, "might want kids": 0, "doesn&rsquo;t want kids": 0
}
df["has_kids_code"] = df.offspring.map(has_kids_mapping)

# creates numerical data describing whether or not an individual wants children
	# 1 = wants or might want children
	# 0 = does not want children
wants_kids_mapping = {
	"has kids, and wants more": 1, "has a kid, and wants more": 1, "doesn&rsquo;t have kids, but wants them": 1,
	"has kids, and might want more": 1, "has a kid, and might want more": 1, "doesn&rsquo;t have kids, but might want them": 1,
	"has kids, but doesn&rsquo;t want more": 0, "has a kid, but doesn&rsquo;t want more": 0, "doesn&rsquo;t have kids, and doesn&rsquo;t want any": 0,
	"has kids": 0, "has a kid": 0, "doesn&rsquo;t have kids": 0,
	"wants kids": 1, "might want kids": 1, "doesn&rsquo;t want kids": 0
}
df["wants_kids_code"] = df.offspring.map(wants_kids_mapping)

# creates numerical data describing an individuals level of education
	# 0 = up to high school
	# 1 = completed high school up to two-year college
	# 2 = completed two-year college up to college/university
	# 3 = completed college/university up to masters program
	# 4 = completed masters program up to law school/med school
	# 5 = completed law school/med school up to ph.d program
	# 6 = completed ph.d
education_mapping = {
	"dropped out of space camp": 0, "working on space camp": 0, "space camp": 0, "graduated from space camp": 0, "dropped out of high school": 0, "working on high school": 0,
	"high school": 1, "graduated from high school": 1, "dropped out of two-year college": 1, "working on two-year college": 1,
	"two-year college": 2, "graduated from two-year college": 2, "dropped out of college/university": 2, "working on college/university": 2,
	"college/university": 3, "graduated from college/university": 3, "dropped out of masters program": 3, "working on masters program": 3,
	"masters program": 4, "graduated from masters program": 4, "dropped out of law school": 4, "dropped out of med school": 4, "working on law school": 4, "working on med school": 4,
	"law school": 5, "med school": 5, "graduated from law school": 5, "graduated from med school": 5, "dropped out of ph.d program": 5, "working on ph.d program": 5,
	"ph.d program": 6, "graduated from ph.d program": 6
}
df["education_code"] = df.education.map(education_mapping)

# creates numerical data describing how important signs are to an individual
	# 0 = it doesn't matter
	# 1 = neutral
	# 2 = it's fun to think about
	# 3 = it matters a lot
sign_importance_mapping = {
	"aries but it doesn&rsquo;t matter": 0, "taurus but it doesn&rsquo;t matter": 0, "gemini but it doesn&rsquo;t matter": 0, "cancer but it doesn&rsquo;t matter": 0, "leo but it doesn&rsquo;t matter": 0, "virgo but it doesn&rsquo;t matter": 0, "libra but it doesn&rsquo;t matter": 0, "scorpio but it doesn&rsquo;t matter": 0, "sagittarius but it doesn&rsquo;t matter": 0, "capricorn but it doesn&rsquo;t matter": 0, "aquarius but it doesn&rsquo;t matter": 0, "pisces but it doesn&rsquo;t matter": 0, 
	"aries": 1, "taurus": 1, "gemini": 1, "cancer": 1, "leo": 1, "virgo": 1, "libra": 1, "scorpio": 1, "sagittarius": 1, "capricorn": 1, "aquarius": 1, "pisces": 1, 
	"aries and it&rsquo;s fun to think about": 2, "taurus and it&rsquo;s fun to think about": 2, "gemini and it&rsquo;s fun to think about": 2, "cancer and it&rsquo;s fun to think about": 2, "leo and it&rsquo;s fun to think about": 2, "virgo and it&rsquo;s fun to think about": 2, "libra and it&rsquo;s fun to think about": 2, "scorpio and it&rsquo;s fun to think about": 2, "sagittarius and it&rsquo;s fun to think about": 2, "capricorn and it&rsquo;s fun to think about": 2, "aquarius and it&rsquo;s fun to think about": 2, "pisces and it&rsquo;s fun to think about": 2, 
	"aries and it matters a lot": 3, "taurus and it matters a lot": 3, "gemini and it matters a lot": 3, "cancer and it matters a lot": 3, "leo and it matters a lot": 3, "virgo and it matters a lot": 3, "libra and it matters a lot": 3, "scorpio and it matters a lot": 3, "sagittarius and it matters a lot": 3, "capricorn and it matters a lot": 3, "aquarius and it matters a lot": 3, "pisces and it matters a lot": 3
}
df["sign_importance_code"] = df.sign.map(sign_importance_mapping)

# creates numerical data describing whether or not an individual has at least one pet
	# 0 = has no pets
	# 1 = has pet(s) 
has_pet_mapping = {
	"dislikes dogs and likes cats": 0, "dislikes dogs and has cats": 1, "likes cats": 0, "has cats": 1,
	"likes dogs and dislikes cats": 0, "has dogs and dislikes cats": 1, "likes dogs": 0, "has dogs": 1,
	"likes dogs and likes cats": 0, "has dogs and has cats": 1, "likes dogs and has cats": 1, "has dogs and likes cats": 1,
	"dislikes dogs and dislikes cats": 0, "dislikes dogs": 0, "dislikes cats": 0
}
df["has_pet_code"] = df.pets.map(has_pet_mapping)

# Below was part of an abandoned experiment with writing grade level data based on essay columns
# Feel free to un-comment below if you would like to view
	# from readability import Readability // documentation found here: https://github.com/cdimascio/py-readability-metrics
	# from html.parser import HTMLParser
	# class MLStripper(HTMLParser):
	#     def __init__(self):
	#         self.reset()
	#         self.strict = False
	#         self.convert_charrefs= True
	#         self.fed = []
	#     def handle_data(self, d):
	#         self.fed.append(d)
	#     def get_data(self):
	#         return ''.join(self.fed)

	# def strip_tags(html):
	#     s = MLStripper()
	#     s.feed(html)
	#     return s.get_data()

	# # Pulled from capstone instructions
	# essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
	# # Removing the NaNs
	# all_essays = df[essay_cols].replace(np.nan, '', regex=True)
	# # Combining the essays
	# all_essays = all_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
	# # Strips HTML from combined essays
	# cleaned_essays = all_essays.apply(strip_tags)
	# # Function to be applied to each combined and cleaned essay to get Flesch Kincaid writing grade level
	# def get_grade_level(essay):
	# 	# Get word count
	# 	word_count = len(essay.split())
	# 	# Ensures minimum word count for grade calculation.
	# 		# This was supposed to be 100, but I received the error "raise ReadabilityException('100 words required.')" until I increased the minimum to 124. Also, output did not match tests I did in Microsoft Word.
	# 	if word_count > 124:
	# 		r = Readability(essay)
	# 		fk = r.flesch_kincaid()
	# 		grade = fk.grade_level
	# 		return grade
	# 	else:
	# 		return 0
	# # Create new column in dataframe for grade level
	# df["writing_level"] = cleaned_essays.apply(get_grade_level)
# I abandoned the above code because I found py-readability-metrics to be inaccurate. It also took a long time to run.

feature_data = df[['age', 'income', 'education_code', 'has_kids_code', 'wants_kids_code', 'has_pet_code', 'drinks_code', 'smokes_code', 'drugs_code', 'sign_importance_code']]
feature_data = feature_data.dropna()



### Normalizing Data

def min_max_normalize(data):
	x = data.values
	min_max_scaler = preprocessing.MinMaxScaler()
	x_scaled = min_max_scaler.fit_transform(x)
	normalized = pd.DataFrame(x_scaled, columns=data.columns)

	return normalized



### Classification with Support Vector Machines

plt.scatter(x = feature_data.age, 
	y = feature_data.education_code, 
	c = feature_data.has_kids_code, 
	cmap = plt.cm.coolwarm, 
	alpha = 0.5)
plt.title("Who Has Kids?")
plt.xlabel("Age")
plt.ylabel("Education")

svm_training_set, svm_validation_set = train_test_split(feature_data, random_state = 1)

# Used to find a reasonable gamma and C value
# largest = {'score': 0, 'gamma': 1, 'C': 1}

# for gamma in range(1, 10):
# 	for C in range(1, 10): 
# 		svm_classifier = SVC(kernel = 'rbf', gamma = gamma, C = C)
# 		svm_classifier.fit(svm_training_set[['age', 'education_code']],svm_training_set['has_kids_code'])
# 		score = svm_classifier.score(svm_validation_set[['age', 'education_code']],svm_validation_set['has_kids_code'])

# 		if(score > largest['score']):
# 			largest['score'] = score
# 			largest['gamma'] = gamma
# 			largest['C'] = C

# print(largest)

svm_classifier = SVC(kernel = 'rbf', gamma = 1, C = 1)
svm_classifier.fit(svm_training_set[['age', 'education_code']],svm_training_set['has_kids_code'])

print("Train score:")
print(svm_classifier.score(svm_training_set[['age', 'education_code']],svm_training_set['has_kids_code']))

print("Test score:")
print(svm_classifier.score(svm_validation_set[['age', 'education_code']],svm_validation_set['has_kids_code']))

plt.show()



## Classification with K-Nearest Neighbors

norm_feature_data = min_max_normalize(feature_data)
k_data = norm_feature_data[['age', 'income', 'education_code', 'wants_kids_code', 'has_pet_code', 'drinks_code', 'smokes_code', 'drugs_code', 'sign_importance_code']]
k_target = norm_feature_data[['has_kids_code']]

k_data = k_data.values
k_target = k_target.values

k_training_data, k_validation_data, k_training_labels, k_validation_labels = train_test_split(
	k_data,
	k_target,
	test_size=0.2,
	random_state = 75
)
# accuracies = []
# for k in range(1, 201):
#   k_classifier = KNeighborsClassifier(n_neighbors = k)
#   k_classifier.fit(k_training_data, np.ravel(k_training_labels,order='C'))
#   accuracies.append(k_classifier.score(k_validation_data, k_validation_labels))

# k_list = range(1, 201)
# plt.plot(k_list, accuracies)
# plt.xlabel("k")
# plt.ylabel("Validation Accuracy")
# plt.title("Has Kid(s) Classifier Accuracy")
# plt.show()

k_classifier = KNeighborsClassifier(n_neighbors = 61)
k_classifier.fit(k_training_data, np.ravel(k_training_labels,order='C'))

print("Train score:")
print(k_classifier.score(k_training_data,k_training_labels))

print("Test score:")
print(k_classifier.score(k_validation_data,k_validation_labels))



### Regression with Multiple Linear Regression

x = feature_data[['income', 'education_code', 'has_kids_code', 'wants_kids_code', 'has_pet_code', 'drinks_code', 'smokes_code', 'drugs_code', 'sign_importance_code']]
y = feature_data[['age']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

mlr = LinearRegression()
mlr.fit(x_train, y_train)

y_predict = mlr.predict(x_test)

plt.scatter(y_test, y_predict, alpha=0.4)
plt.xlabel("Age")
plt.ylabel("Predictions")
plt.title("Actual Age vs Predicted Age")
plt.show()

print("Train score:")
print(mlr.score(x_train, y_train))

print("Test score:")
print(mlr.score(x_test, y_test))

print(mlr.coef_)



### Regression with Linear Regression

age = feature_data[['age']]
age = age.values
education = feature_data[['education_code']]
education = education.values

lr_x_train, lr_x_test, lr_y_train, lr_y_test = train_test_split(age, education, train_size = 0.8, test_size = 0.2, random_state=6)

line_fitter = LinearRegression()
line_fitter.fit(lr_x_train, lr_y_train)

ed_predict = line_fitter.predict(lr_x_test)

plt.plot(lr_x_train, lr_y_train, 'o')
plt.plot(lr_x_test, ed_predict)
plt.xlabel('Age')
plt.ylabel('Education Level')
plt.title('Age vs Education')
plt.show()

print("Train score:")
print(line_fitter.score(lr_x_train, lr_y_train))

print("Test score:")
print(line_fitter.score(lr_x_test, lr_y_test))







