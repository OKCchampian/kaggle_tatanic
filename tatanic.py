import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

train_data = pd.read_csv('/Users/wangxiyao/Desktop/titanic/train.csv')
test_data = pd.read_csv('/Users/wangxiyao/Desktop/titanic/test.csv')
# train_data.info()
# test_data.info()
train_data['Survived'].value_counts()
train_data.groupby(['Sex', 'Survived'])['Survived'].count()
train_data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar()
train_data[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar(color=['g','b'])
train_data[['Sex','Pclass','Survived']].groupby(['Pclass','Sex']).mean().plot.bar()

f, ax = plt.subplots(1, 2, figsize=(18, 8))
sns.violinplot('Pclass', 'Age', hue='Survived', data=train_data, split=True, ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot("Sex","Age", hue="Survived", data=train_data,split=True,ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))

sns.countplot('Embarked',hue='Survived',data=train_data)
plt.title('Embarked and Survived')

f, ax=plt.subplots(1,2,figsize=(18,8))
train_data[['Parch','Survived']].groupby(['Parch']).mean().plot.bar(ax=ax[0])
ax[0].set_title('Parch and Survived')
train_data[['SibSp','Survived']].groupby(['SibSp']).mean().plot.bar(ax=ax[1])
ax[1].set_title('SibSp and Survived')

train_data_org = pd.read_csv('/Users/wangxiyao/Desktop/titanic/train.csv')
test_data_org = pd.read_csv('/Users/wangxiyao/Desktop/titanic/test.csv')
conbined_train_test = train_data_org.append(test_data_org, sort = False)

conbined_train_test['Cabin']=conbined_train_test['Cabin'].fillna('U')

conbined_train_test['Embarked'].value_counts()
conbined_train_test['Embarked']=conbined_train_test['Embarked'].fillna('S')

# conbined_train_test[conbined_train_test['Fare'].isnull()]
conbined_train_test['Fare']=conbined_train_test['Fare'].fillna(conbined_train_test[(conbined_train_test['Pclass']==3)&(conbined_train_test['Embarked']=='C')&(conbined_train_test['Cabin']=='U')]['Fare'].mean())
conbined_train_test['Deck']=conbined_train_test['Cabin'].map(lambda x:x[0])

conbined_train_test['Title']=conbined_train_test['Name'].map(lambda x:x.split(',')[1].split('.')[0].strip())
#查看title数据分布
conbined_train_test['Title'].value_counts()
TitleDict = {}
TitleDict['Mr']='Mr'
TitleDict['Mlle']='Miss'
TitleDict['Miss']='Miss'
TitleDict['Master']='Master'
TitleDict['Jonkheer']='Master'
TitleDict['Mme']='Mrs'
TitleDict['Ms']='Mrs'
TitleDict['Mrs']='Mrs'
TitleDict['Don']='Royalty'
TitleDict['Sir']='Royalty'
TitleDict['the Countess']='Royalty'
TitleDict['Dona']='Royalty'
TitleDict['Lady']='Royalty'
TitleDict['Capt']='Officer'
TitleDict['Col']='Officer'
TitleDict['Major']='Officer'
TitleDict['Dr']='Officer'
TitleDict['Rev']='Officer'

conbined_train_test['Title']=conbined_train_test['Title'].map(TitleDict)
conbined_train_test['Title'].value_counts()

conbined_train_test['Family_num'] = conbined_train_test['Parch'] + conbined_train_test['SibSp'] + 1
def family_size(Family_num):
    if Family_num==1:
        return 0
    elif Family_num>=2 and Family_num<=4:
        return 1
    else:
        return 2
conbined_train_test['Family_size'] = conbined_train_test['Family_num'].map(family_size)

conbined_train_test['Family_size'].value_counts()
conbined_train_test['Deck']=conbined_train_test['Cabin'].map(lambda x:x[0])
TickCountDict={}
TickCountDict=conbined_train_test['Ticket'].value_counts()
TickCountDict.head()
conbined_train_test['TickCot']=conbined_train_test['Ticket'].map(TickCountDict)
def TickCountGroup(num):
    if (num>=2)&(num<=4):
        return 0
    elif (num==1)|((num>=5)&(num<=8)):
        return 1
    else :
        return 2
#得到各位乘客TickGroup的类别
conbined_train_test['TickGroup']=conbined_train_test['TickCot'].map(TickCountGroup)
# 使用RF填充年龄
AgePre = conbined_train_test[['Age','Parch','Pclass','SibSp','Title','Family_num','TickCot']]
AgePre = pd.get_dummies(AgePre)
ParAge = pd.get_dummies(AgePre['Parch'],prefix='Parch')
SibAge = pd.get_dummies(AgePre['SibSp'],prefix='SibSp')
PclAge = pd.get_dummies(AgePre['Pclass'],prefix='Pclass')
# AgeCorrDf=pd.DataFrame()
AgeCorrDf=AgePre.corr()
AgeCorrDf['Age'].sort_values()

AgePre=pd.concat([AgePre,ParAge,SibAge,PclAge],axis=1)
AgePre.head()

Ageknow = AgePre[AgePre['Age'].notnull()]
Ageunknow = AgePre[AgePre['Age'].isnull()]
Ageknow_x = Ageknow.drop(['Age'], axis=1)
Ageknow_y = Ageknow['Age']
Ageunknow_x = Ageunknow.drop(['Age'], axis=1)

from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(random_state=None,n_estimators=600,n_jobs=-1)
rfr.fit(Ageknow_x, Ageknow_y)
rfr.score(Ageknow_x, Ageknow_y)
Ageunknow_y = rfr.predict(Ageunknow_x)
conbined_train_test.loc[conbined_train_test['Age'].isnull(), ['Age']] = Ageunknow_y
# 同性相关分析
conbined_train_test['Surname'] = conbined_train_test['Name'].map(lambda x:x.split(',')[0].strip())
Surnamedict = conbined_train_test['Surname'].value_counts()
conbined_train_test['Surname_num'] = conbined_train_test['Surname'].map(Surnamedict)
# 修改异常值
male = conbined_train_test[(conbined_train_test['Sex'] == 'male')&(conbined_train_test['Age']>12)&(conbined_train_test['Family_num']>=2)]
female_child = conbined_train_test[((conbined_train_test['Sex'] == 'female')|(conbined_train_test['Age']<=12))&(conbined_train_test['Family_num']>=2)]

MSurNamDf=male['Survived'].groupby(male['Surname']).mean()
MSurNamDf.head()
MSurNamDf.value_counts()

MSurNamDf = male['Survived'].groupby(male['Surname']).mean()
MSurNamDf.value_counts()
MSurNamDict = MSurNamDf[MSurNamDf.values == 1].index
FCSurNamDf = female_child['Survived'].groupby(female_child['Surname']).mean()
FCSurNamDf.value_counts()
FCSurNamDict = FCSurNamDf[FCSurNamDf.values == 0].index


conbined_train_test.loc[(conbined_train_test['Survived'].isnull())&(conbined_train_test['Surname'].isin(MSurNamDict))&(conbined_train_test['Sex']=='male'),'Age']=5
conbined_train_test.loc[(conbined_train_test['Survived'].isnull())&(conbined_train_test['Surname'].isin(MSurNamDict))&(conbined_train_test['Sex']=='male'),'Sex']='female'
conbined_train_test.loc[(conbined_train_test['Survived'].isnull())&(conbined_train_test['Surname'].isin(FCSurNamDict))&((conbined_train_test['Sex']=='female')|(conbined_train_test['Age']<=12)),'Age']=60
conbined_train_test.loc[(conbined_train_test['Survived'].isnull())&(conbined_train_test['Surname'].isin(FCSurNamDict))&((conbined_train_test['Sex']=='female')|(conbined_train_test['Age']<=12)),'Sex']='male'



data=conbined_train_test.drop(['Cabin','Name','Ticket','PassengerId','Surname','Surname_num'],axis=1)
#查看各特征与标签的相关性
# corrDf=pd.DataFrame()
# corrDf=data.corr()
# corrDf['Survived'].sort_values(ascending=True)
data=data.drop(['Family_num','SibSp','TickCot','Parch'],axis=1)
data=pd.get_dummies(data)
PclassDf=pd.get_dummies(conbined_train_test['Pclass'],prefix='Pclass')
TickGroupDf=pd.get_dummies(conbined_train_test['TickGroup'],prefix='TickGroup')
familySizeDf=pd.get_dummies(conbined_train_test['Family_size'],prefix='Family_size')

data=pd.concat([data,PclassDf,TickGroupDf,familySizeDf],axis=1)

data.info()


experData=fullSel[fullSel['Survived'].notnull()]
preData=fullSel[fullSel['Survived'].isnull()]

experData_X=experData.drop('Survived',axis=1)
experData_y=experData['Survived']
preData_X=preData.drop('Survived',axis=1)

experData=data[data['Survived'].notnull()]
preData=data[data['Survived'].isnull()]

experData_X=experData.drop('Survived',axis=1)
experData_y=experData['Survived']
preData_X=preData.drop('Survived',axis=1)

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier,ExtraTreesClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,cross_val_score,StratifiedKFold

#设置kfold，交叉采样法拆分数据集
kfold=StratifiedKFold(n_splits=10)
classifiers=[]
classifiers.append(SVC())
classifiers.append(DecisionTreeClassifier())
classifiers.append(RandomForestClassifier())
classifiers.append(ExtraTreesClassifier())
classifiers.append(GradientBoostingClassifier())
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression())
classifiers.append(LinearDiscriminantAnalysis())
cv_results=[]
for classifier in classifiers:
    cv_results.append(cross_val_score(classifier,experData_X,experData_y,
                                      scoring='accuracy',cv=kfold,n_jobs=-1))
cv_means=[]
cv_std=[]
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
cvResDf=pd.DataFrame({'cv_mean':cv_means,
                     'cv_std':cv_std,
                     'algorithm':['SVC','DecisionTreeCla','RandomForestCla','ExtraTreesCla',
                                  'GradientBoostingCla','KNN','LR','LinearDiscrimiAna']})
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1]
              }
modelgsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold,
                                     scoring="accuracy", n_jobs= -1, verbose = 1)
modelgsGBC.fit(experData_X,experData_y)

#LogisticRegression模型
modelLR=LogisticRegression()
LR_param_grid = {'C' : [1,2,3],
                'penalty':['l1','l2']}
modelgsLR = GridSearchCV(modelLR,param_grid = LR_param_grid, cv=kfold,
                                     scoring="accuracy", n_jobs= -1, verbose = 1)
modelgsLR.fit(experData_X,experData_y)

# print('modelgsGBC模型得分为：%.3f'%modelgsGBC.best_score_)
# modelgsLR模型
# print('modelgsLR模型得分为：%.3f'%modelgsLR.best_score_)

# 选择LR
GBCpreData_y=modelgsLR.predict(preData_X)
GBCpreData_y=GBCpreData_y.astype(int)
#导出预测结果
GBCpreResultDf=pd.DataFrame()
GBCpreResultDf['PassengerId']=conbined_train_test['PassengerId'][conbined_train_test['Survived'].isnull()]
GBCpreResultDf['Survived']=GBCpreData_y
GBCpreResultDf
#将预测结果导出为csv文件
GBCpreResultDf.to_csv('/Users/wangxiyao/Desktop/TitanicLRmodle.csv',index=False)
