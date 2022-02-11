import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes =True)
#
dia=pd.read_csv("diabetes.csv")
diab=pd.read_csv("diabetes.csv")
#
dia1 = dia[dia.Outcome==1]
dia0 = dia[dia.Outcome==0]
diab.describe()
#
## Creating a dataset called 'dia' from original dataset 'diab' with excludes all rows with have zeros only for Glucose, BP, Skinthickness, Insulin and BMI, as other columns can contain Zero values.
drop_Glu=diab.index[diab.Glucose == 0].tolist()
drop_BP=diab.index[diab.BloodPressure == 0].tolist()
drop_Skin = diab.index[diab.SkinThickness==0].tolist()
drop_Ins = diab.index[diab.Insulin==0].tolist()
drop_BMI = diab.index[diab.BMI==0].tolist()
c=drop_Glu+drop_BP+drop_Skin+drop_Ins+drop_BMI
dia=diab.drop(diab.index[c])
#
## creating count plot with title using seaborn
sns.countplot(x=dia.Outcome)
plt.title("Count Plot for Outcome")
#
# Computing the %age of diabetic and non-diabetic in the sample
Out0=len(dia[dia.Outcome==1])
Out1=len(dia[dia.Outcome==0])
Total=Out0+Out1
PC_of_1 = Out1*100/Total
PC_of_0 = Out0*100/Total
PC_of_1, PC_of_0

#PREGNANCY
## Creating 3 subplots - 1st for histogram, 2nd for histogram segmented by Outcome and 3rd for representing same segmentation using boxplot
plt.figure(figsize=(20, 6))
plt.subplot(1,3,1)
sns.set_style("dark")
plt.title("Histogram for Pregnancies")
sns.distplot(dia.Pregnancies,kde=False)
plt.subplot(1,3,2)
sns.distplot(dia0.Pregnancies,kde=False,color="Blue", label="Preg for Outome=0")
sns.distplot(dia1.Pregnancies,kde=False,color = "Gold", label = "Preg for Outcome=1")
plt.title("Histograms for Preg by Outcome")
plt.legend()
plt.subplot(1,3,3)
sns.boxplot(x=dia.Outcome,y=dia.Pregnancies)
plt.title("Boxplot for Preg by Outcome")
plt.savefig("pregnancy.png")
#
#GLUCOSE
plt.figure(figsize=(20, 6))
plt.subplot(1,3,1)
plt.title("Histogram for Glucose")
sns.distplot(dia.Glucose, kde=False)
plt.subplot(1,3,2)

distplot(dia0.Glucose,kde=False,color="Gold", label="Gluc for Outcome=0")
sns.distplot(dia1.Glucose, kde=False, color="Blue", label = "Gloc for Outcome=1")
plt.title("Histograms for Glucose by Outcome")
plt.legend()
plt.subplot(1,3,3)
sns.boxplot(x=dia.Outcome,y=dia.Glucose)
plt.title("Boxplot for Glucose by Outcome")
plt.savefig("glucose.png")
#
#BLOOD PRESSURE
plt.figure(figsize=(20, 6))
plt.subplot(1,3,1)
sns.distplot(dia.BloodPressure, kde=False)
plt.title("Histogram for Blood Pressure")
plt.subplot(1,3,2)
sns.distplot(dia0.BloodPressure,kde=False,color="Gold",label="BP for Outcome=0")
sns.distplot(dia1.BloodPressure,kde=False, color="Blue", label="BP for Outcome=1")
plt.legend()
plt.title("Histogram of Blood Pressure by Outcome")
plt.subplot(1,3,3)
sns.boxplot(x=dia.Outcome,y=dia.BloodPressure)
plt.title("Boxplot of BP by Outcome")
plt.savefig("bloodpressure.png")
#
#SKIN THICKNESS
plt.figure(figsize=(20, 6))
plt.subplot(1,3,1)
sns.distplot(dia.SkinThickness, kde=False)
plt.title("Histogram for Skin Thickness")
plt.subplot(1,3,2)
sns.distplot(dia0.SkinThickness, kde=False, color="Gold", label="SkinThick for Outcome=0")
sns.distplot(dia1.SkinThickness, kde=False, color="Blue", label="SkinThick for Outcome=1")
plt.legend()
plt.title("Histogram for SkinThickness by Outcome")
plt.subplot(1,3,3)
sns.boxplot(x=dia.Outcome, y=dia.SkinThickness)
plt.title("Boxplot of SkinThickness by Outcome")
plt.savefig("skinthickness.png")
#
#INSULIN
plt.figure(figsize=(20, 6))
plt.subplot(1,3,1)
sns.distplot(dia.Insulin,kde=False)
plt.title("Histogram of Insulin")
plt.subplot(1,3,2)
sns.distplot(dia0.Insulin,kde=False, color="Gold", label="Insulin for Outcome=0")
sns.distplot(dia1.Insulin,kde=False, color="Blue", label="Insuline for Outcome=1")
plt.title("Histogram for Insulin by Outcome")
plt.legend()
plt.subplot(1,3,3)
sns.boxplot(x=dia.Outcome, y=dia.Insulin)
plt.title("Boxplot for Insulin by Outcome")
plt.savefig("insulin.png")
#
#BODY-MASS-INDEX
plt.figure(figsize=(20, 6))
plt.subplot(1,3,1)
sns.distplot(dia.BMI, kde=False)
plt.title("Histogram for BMI")
plt.subplot(1,3,2)
sns.distplot(dia0.BMI, kde=False,color="Gold", label="BMI for Outcome=0")
sns.distplot(dia1.BMI, kde=False, color="Blue", label="BMI for Outcome=1")
plt.legend()
plt.title("Histogram for BMI by Outcome")
plt.subplot(1,3,3)
sns.boxplot(x=dia.Outcome, y=dia.BMI)
plt.title("Boxplot for BMI by Outcome")
plt.savefig("BMI.png")
#
#DIABETES PEDIGREE FUNCTION
plt.figure(figsize=(20, 6))
plt.subplot(1,3,1)
sns.distplot(dia.DiabetesPedigreeFunction,kde=False)
plt.title("Histogram for Diabetes Pedigree Function")
plt.subplot(1,3,2)
sns.distplot(dia0.DiabetesPedigreeFunction, kde=False, color="Gold", label="PedFunction for Outcome=0")
sns.distplot(dia1.DiabetesPedigreeFunction, kde=False, color="Blue", label="PedFunction for Outcome=1")
plt.legend()
plt.title("Histogram for DiabetesPedigreeFunction by Outcome")
plt.subplot(1,3,3)
sns.boxplot(x=dia.Outcome, y=dia.DiabetesPedigreeFunction)
plt.title("Boxplot for DiabetesPedigreeFunction by Outcome")
plt.savefig("diapedifun.png")
#
#AGE
plt.figure(figsize=(20, 6))
plt.subplot(1,3,1)
sns.distplot(dia.Age,kde=False)
plt.title("Histogram for Age")
plt.subplot(1,3,2)
sns.distplot(dia0.Age,kde=False,color="Gold", label="Age for Outcome=0")
sns.distplot(dia1.Age,kde=False, color="Blue", label="Age for Outcome=1")
plt.legend()
plt.title("Histogram for Age by Outcome")
plt.subplot(1,3,3)
sns.boxplot(x=dia.Outcome,y=dia.Age)
plt.title("Boxplot for Age by Outcome")
plt.savefig("age.png")
