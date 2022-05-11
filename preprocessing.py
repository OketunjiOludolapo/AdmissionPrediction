import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
class preprocessing:
    def __init__(self,df):
        self.df=df
    
    def read(self):
        c=pd.read_csv(self.df)
        c=c.dropna()
        return c
    
    def drop(self,m,columns):
        m.drop(columns=columns,inplace=True)
        return m
    
    def split(self,v,target,test_size,state=None):
        X=v.drop(columns=target)
        y=v[target]
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size)
        return X_train,X_test,y_train,y_test

    def standard_scaling(self,x):
        X_train=x[0]
        X_test=x[1]
        y_train=x[2]
        y_test=x[3]
        scale=StandardScaler()
        arr=scale.fit_transform(X_train)
        arr_test=scale.transform(X_test)
        return arr,arr_test,y_train,y_test
    
    def model(self,c):
        linear=LinearRegression()
        linear.fit(c[0],c[2])
        score=linear.score(c[1],c[3])
        return score,linear
    
    def save_to_model(self,model,filename):
        import pickle
        return pickle.dump(model[1],open(f"{filename}.pickle","wb"))
        




process=preprocessing("Admission_Prediction.csv")
data=process.read()
drop=process.drop(data, columns="Serial No.")
split=process.split(drop,"Chance of Admit",0.3,6)
scaled=process.standard_scaling(split)
model=process.model(scaled)
print(process.save_to_model(model,"model"))