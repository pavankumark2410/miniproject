from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as nm
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier 
import matplotlib.pyplot as plt
import base64
from io import BytesIO
def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buffer.close()
    return graph
def get_plot(x,y):
    plt.switch_backend('AGG')
    plt.figure(figsize=(5,5))
    plt.title('Deaths VS COUNT')
    plt.bar(x,y)
    #plt.xticks(rotation=45)
    plt.xlabel('Death')
    plt.ylabel('COUNT')
    plt.tight_layout()
    graph = get_graph()
    return graph

def algorithm(request):
    if request.method=='POST':
        age=request.POST.get('age')
        gender=request.POST.get('gender')
        chestpain=request.POST.get('chestpain')
        restingbp =request.POST.get('restingbp')
        cholestrollevel=request.POST.get('cholestrollevel')
        fastingbs =request.POST.get('fastingbs')
        restingecg =request.POST.get('restingecg')
        maxhrlevel =request.POST.get('maxhrlevel')
        Agina =request.POST.get('Agina')
        oldpeck =request.POST.get('oldpeck')
        stslope=request.POST.get('stslope')
        if not (age) :
            age=0
        if not (gender) :
            gender=0
        if not (chestpain) :
            chestpain=0
        if not (restingbp) :
            restingbp=0
        if not (cholestrollevel) :
            cholestrollevel=0
        if not (fastingbs) :
            fastingbs=0
        if not (restingecg) :
            restingecg=0
        if not (maxhrlevel) :
            maxhrlevel=0
        if not (Agina) :
            Agina=0
        if not (oldpeck) :
            oldpeck=0
        if not (stslope) :
            stslope=0
        print("age: {}\ngender:{}\nchestpain:{}\nrestingbp:{}\ncholestrollevel:{}\nfastinbs:{}\nrestingecg:{}\nmaxhrlevel:{}\nAgina:{}\noldpeck:{}\nstslope:{}".format(
            age,gender,chestpain,restingbp,cholestrollevel,fastingbs,restingecg,maxhrlevel,Agina,
            oldpeck,stslope
            ))
        df=pd.read_csv("cia3/csvfile/heart.csv")
        labelencoder=LabelEncoder()
        col=['Sex']
        df['Sex']=labelencoder.fit_transform(df['Sex'])
        df['ChestPainType']=labelencoder.fit_transform(df['ChestPainType'])
        df['ExerciseAngina']=labelencoder.fit_transform(df['ExerciseAngina'])
        df['ST_Slope']=labelencoder.fit_transform(df['ST_Slope'])
        df['RestingECG']=labelencoder.fit_transform(df['RestingECG'])
        col=['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope']
        X = df[col]
        Y = df['HeartDisease']
        X_train,X_test,y_train,y_test = train_test_split(X, Y, random_state=0)
        random_decision_tree= RandomForestClassifier(n_estimators= 10, criterion="entropy")
        random_decision_tree.fit(X,Y)
        """
        x:
        Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina, Oldpeak,ST_Slope
        y:
        HeartDisease
        """
        features = nm.array([[float(age),float(gender),float(chestpain),float(restingbp),float(cholestrollevel),float(fastingbs),float(restingecg),float(maxhrlevel),float(Agina),
            float(oldpeck),float(stslope)]])
        preds = random_decision_tree.predict(features)
        random_decision_prediction = random_decision_tree.predict(X_test)
        acc_decision_tree = round(random_decision_tree.score(X_train, y_train) * 100,8)
        acc_decision_tree
        pred=preds[0]
        print(type(pred))
        
        if pred==0:
            str="The person is free from heart failure"
        if pred==1:
            str="The person might suffer from heart failure"
       
        #chart = get_plot()
        gendercount=list()
        gender=list()
        x=df['HeartDisease'].groupby(df['HeartDisease'])
        x=df[df['HeartDisease'] ==1]['HeartDisease'].count()
        gender.append("TOTAL HEART FAILURE")
        gendercount.append(x)
        x=df[df['HeartDisease'] ==0]['HeartDisease'].count()
        gender.append("NO HEART FAIURE")
        gendercount.append(x)
        chart = get_plot(gender,gendercount)
        context={'pred':preds[0],'str':str,'chart': chart}
        
        
       
    
        return render(request,'index.html',context)
    return render(request,'index.html')


    