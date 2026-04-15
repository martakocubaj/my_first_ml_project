import matplotlib.pyplot as plt
import pandas as pd
import warnings
import seaborn as sns


## Age categorization

def age_cat(years):
    if years <= 30:
        return '0-30'
    elif years > 30 and years <= 45:
        return '30-45'
    elif years > 45 and years <= 60:
        return '45-60'
    elif years > 60:
        return '60+'
    
    

## This function plots distribution and count of a variable by category

def bi_cat_countplot(df, column, hue_column):
    unique_hue_values = df[hue_column].unique()
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(16, 7) 

 
    pltname = f'Normal distribution: {column}'
    proportions = df.groupby(hue_column)[column].value_counts(normalize=True)
    proportions = (proportions * 100).round(2)
    ax1 = proportions.unstack(hue_column).sort_values(
        by=unique_hue_values[0], ascending=False
    ).plot.bar(ax=axes[0], title=pltname)

    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f%%', padding=3)

 
    pltname = f'Count: {column}'
    counts = df.groupby(hue_column)[column].value_counts()
    ax2 = counts.unstack(hue_column).sort_values(
        by=unique_hue_values[0], ascending=False
    ).plot.bar(ax=axes[1], title=pltname)

    for container in ax2.containers:
        ax2.bar_label(container, padding=3)

   
    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0) 
        ax.set_xlabel('') 
    plt.tight_layout() 
    plt.show()

    
    

def uni_cat_target_compare(df, column):
    bi_cat_countplot(df, column, hue_column='y' )

    
    
    

## This function compares distributions and counts for two groups of clients    
    
def bi_countplot_target(df0, df1, column, hue_column):
  pltname = 'Client subscribed to a term deposit'
  print(pltname.upper())
  bi_cat_countplot(df1, column, hue_column)
  plt.show()

  pltname = 'Client did not subscribe to a term deposit'
  print(pltname.upper())
  bi_cat_countplot(df0, column, hue_column)
  plt.show()


    
    
    
    
## For outlier rate:   
    
def outlier_range(dataset,column):
    Q1 = dataset[column].quantile(0.25)
    Q3 = dataset[column].quantile(0.75)
    IQR = Q3 - Q1
    Min_value = (Q1 - 1.5 * IQR)
    Max_value = (Q3 + 1.5 * IQR)
    return Max_value



def kde_no_outliers(df0, df1, Max_value0, Max_value1, column):
  plt.figure(figsize = (14,6))
  sns.kdeplot(df1[df1[column] <= Max_value1][column],label = 'Subscribed deposit')
  sns.kdeplot(df0[df0[column] <= Max_value0][column],label = 'Not subscribed deposit')
  plt.ticklabel_format(style='plain', axis='x')
  plt.xticks(rotation = 45)
  plt.legend()
  plt.show()



## For outlier vizualization:

def dist_box(dataset, column):
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")

      plt.figure(figsize=(16,6))

      plt.subplot(1,2,1)
      sns.distplot(dataset[column], color = 'purple')
      pltname = 'Distribution plot for ' + column
      plt.ticklabel_format(style='plain', axis='x')
      plt.title(pltname)

      plt.subplot(1,2,2)
      red_diamond = dict(markerfacecolor='r', marker='D')
      sns.boxplot(y = column, data = dataset, flierprops = red_diamond)
      pltname = 'Boxplot for ' + column
      plt.title(pltname)

      plt.show()