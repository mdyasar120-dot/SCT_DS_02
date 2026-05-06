import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Avoid scientific notation
plt.rcParams['axes.formatter.useoffset'] = False

df = pd.read_csv("task2.csv")

# Data Cleaning
df.columns = df.columns.str.lower()
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

if 'cabin' in df.columns:
    df.drop(columns=['cabin'], inplace=True)
df.drop_duplicates(inplace=True)


# Function to Add Counts
def add_labels(ax):
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height):
            ax.annotate(f'{int(height)}',
                        (p.get_x() + p.get_width()/2., height),
                        ha='center', va='bottom', fontsize=9)


# Survival Count
plt.figure()
ax = sns.countplot(x='survived', data=df)
add_labels(ax)
plt.title("Survival Count")
plt.ylabel("Count")
plt.show()


# Survival by Gender
plt.figure()
ax = sns.countplot(x='sex', hue='survived', data=df)
add_labels(ax)
plt.title("Survival by Gender")
plt.ylabel("Count")
plt.show()


# Survival by Class
plt.figure()
ax = sns.countplot(x='pclass', hue='survived', data=df)
add_labels(ax)
plt.title("Survival by Class")
plt.ylabel("Count")
plt.show()


# Age Distribution
plt.figure()
sns.histplot(df['age'], bins=30, kde=True)
plt.ticklabel_format(style='plain', axis='y')
plt.title("Age Distribution")
plt.ylabel("Count")
plt.show()


# Fare Distribution
plt.figure()
sns.histplot(df['fare'], bins=30, kde=True)
plt.ticklabel_format(style='plain', axis='y')
plt.title("Fare Distribution")
plt.ylabel("Count")
plt.show()


# Relationships

# Age vs Survival
plt.figure()
sns.boxplot(x='survived', y='age', data=df)
plt.title("Age vs Survival")
plt.show()


# Fare vs Survival
plt.figure()
sns.boxplot(x='survived', y='fare', data=df)
plt.title("Fare vs Survival")
plt.show()


# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# Pairplot
sns.pairplot(df[['survived','age','fare','pclass']], hue='survived')
plt.show()


# Statistical Analysis
print("\nAverage Fare by Survival:")
print(df.groupby('survived')['fare'].mean())

print("\nSurvival Rate by Class:")
print(df.groupby('pclass')['survived'].mean())

print("\nSurvival Rate by Gender:")
print(df.groupby('sex')['survived'].mean())


# Final Insights
print("\nKey Insights:")
print("1. Females have higher survival rate than males.")
print("2. 1st class passengers survived more than others.")
print("3. Passengers who paid higher fare had better survival chances.")
print("4. Younger passengers had slightly higher survival probability.")
print("5. Strong relationship exists between class, fare and survival.")