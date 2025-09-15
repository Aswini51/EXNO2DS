# EXNO2DS
# AIM:
To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("/content/titanic_dataset.csv")
df
```
<img width="1187" height="428" alt="Screenshot 2025-09-15 192115" src="https://github.com/user-attachments/assets/7f39f2eb-3fae-45a5-b539-29c4c0f71987" />

```
df.head(10)
```
<img width="1171" height="367" alt="Screenshot 2025-09-15 192128" src="https://github.com/user-attachments/assets/3d430068-1924-44d5-aba0-dfecc3813f26" />

```
df.tail(10)
```
<img width="1130" height="360" alt="Screenshot 2025-09-15 192158" src="https://github.com/user-attachments/assets/cd2641c5-1ab6-4995-b33e-58529f879659" />

```
df.info()
```
<img width="370" height="339" alt="Screenshot 2025-09-15 192221" src="https://github.com/user-attachments/assets/6bb4d832-b5f3-438b-a5ac-a554109742c7" />

```
df.describe()
```
<img width="735" height="294" alt="Screenshot 2025-09-15 192239" src="https://github.com/user-attachments/assets/c5af44b6-7c29-4750-824a-fa2676e5bc6a" />

```
df.shape
```
<img width="136" height="38" alt="Screenshot 2025-09-15 192425" src="https://github.com/user-attachments/assets/b013a3a4-2caf-48d5-b769-51a67cc0815b" />

```
df.set_index("PassengerId",inplace=True)
df["Survived"].value_counts()
```
<img width="194" height="181" alt="Screenshot 2025-09-15 192433" src="https://github.com/user-attachments/assets/30cfe4ce-d90d-4d89-9e24-ad2578c97eaf" />

```
per=(df["Survived"].value_counts()/df.shape[0]*100).round(2)
per
```
<img width="192" height="172" alt="Screenshot 2025-09-15 192441" src="https://github.com/user-attachments/assets/65947e6b-d052-4d87-8411-fad7e9d7dfda" />

```
sns.countplot(data=df,x="Survived")
```
<img width="649" height="461" alt="Screenshot 2025-09-15 192452" src="https://github.com/user-attachments/assets/d7255d65-363d-422c-a1db-8801a762f2bb" />

```
df.Pclass.unique()
```
<img width="175" height="36" alt="Screenshot 2025-09-15 192500" src="https://github.com/user-attachments/assets/37b2d453-6835-49df-8f84-18799ddef0b8" />

```
df.rename(columns={'Sex':'Gender'},inplace=True)
df
```
<img width="1160" height="446" alt="Screenshot 2025-09-15 192518" src="https://github.com/user-attachments/assets/3e696a64-b268-4d83-8287-19289f3f0294" />

```
sns.countplot(data=df,x="Survived")
```
<img width="629" height="461" alt="Screenshot 2025-09-15 192534" src="https://github.com/user-attachments/assets/c6921ea7-70d7-4b57-b5d9-df04aed7a9bc" />

```
sns.catplot(x="Gender",col="Survived",kind="count",data=df,height=5)
```
<img width="1088" height="522" alt="Screenshot 2025-09-15 192602" src="https://github.com/user-attachments/assets/082b9003-e145-43d5-9e26-3943f4e02f4f" />

```
sns.catplot(x="Survived",hue="Gender",data=df,kind="count")
```
<img width="640" height="518" alt="Screenshot 2025-09-15 192618" src="https://github.com/user-attachments/assets/01b1468f-59f6-49b6-a1b1-6f05848d94dd" />

```
df.boxplot(column="Age",by="Survived")
```
<img width="642" height="492" alt="Screenshot 2025-09-15 192633" src="https://github.com/user-attachments/assets/473e05f0-c8be-432f-8b5c-bda6e27f8385" />

```
sns.scatterplot(x=df["Age"],y=df["Fare"])
```
<img width="642" height="466" alt="Screenshot 2025-09-15 192649" src="https://github.com/user-attachments/assets/109a9731-786f-4dcd-876a-98e8c93cc599" />

```
sns.jointplot(x="Age",y="Fare",data=df)
```
<img width="643" height="622" alt="Screenshot 2025-09-15 192709" src="https://github.com/user-attachments/assets/350f9960-6409-429d-a6e6-e075fc938b7a" />

```
fig,ax1 = plt.subplots(figsize=(8,5))
sns.boxplot(ax=ax1, x="Pclass",y="Age",hue="Gender",data=df)
```
<img width="737" height="481" alt="Screenshot 2025-09-15 192739" src="https://github.com/user-attachments/assets/ff0ed831-44b8-40c7-b242-62fc634ecd04" />

```
sns.catplot(data=df,col="Survived",x="Gender",hue="Pclass",kind="count")
```
<img width="1120" height="522" alt="Screenshot 2025-09-15 192754" src="https://github.com/user-attachments/assets/2da8ea25-0426-4577-9045-15240245d274" />

```
## Co-relation
corr=df.corr()
sns.heatmap(corr,annot=True)
```
<img width="642" height="434" alt="Screenshot 2025-09-15 193049" src="https://github.com/user-attachments/assets/5b354a17-d413-4a0f-8f43-2a4bebb20651" />

```
sns.pairplot(df)
```
<img width="1310" height="815" alt="Screenshot 2025-09-15 193401" src="https://github.com/user-attachments/assets/77bb31c4-2cb1-450c-a314-e130b2b25df8" />
<img width="1306" height="448" alt="Screenshot 2025-09-15 193503" src="https://github.com/user-attachments/assets/a348eab7-2fb9-404b-a319-d696139cbe3b" />

# SUMMERY:
The Titanic dataset includes information about passengers such as age, gender, class, fare, and survival status. About 38% of the passengers survived while 62% did not. Survival chances were higher for women compared to men because of the “women and children first” rule. Passengers in first class had better chances of survival than those in third class. Younger passengers, especially children, were more likely to survive. Wealthier passengers who paid higher fares also had higher survival chances. Overall, survival showed strong correlation with gender, passenger class, and fare.
# RESULT:
Thus, the Exploratory Data Analysis on the given data set was performed successfully.
