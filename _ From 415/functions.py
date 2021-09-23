def unistats(df):
  import pandas as pd
  output_df = pd.DataFrame(columns=['Count', 'Missing', 'Unique', 'Dtype', 'Numeric', 'Mode', 'Mean', 'Min', '25%', 'Median', '75%', 'Max', 'Std', 'Skew', 'Kurt'])

  for col in df:
    if pd.api.types.is_numeric_dtype(df[col]):
      output_df.loc[col] = [df[col].count(), df[col].isnull().sum(), df[col].nunique(), df[col].dtype, pd.api.types.is_numeric_dtype(df[col]), 
                            df[col].mode().values[0], df[col].mean(), df[col].min(), df[col].quantile(0.25), df[col].median(), df[col].quantile(0.75),
                            df[col].max(), df[col].std(), df[col].skew(), df[col].kurt()]
    else:
      output_df.loc[col] = [df[col].count(), df[col].isnull().sum(), df[col].nunique(), df[col].dtype, pd.api.types.is_numeric_dtype(df[col]), 
                            df[col].mode().values[0], '-', '-', '-', '-', '-', '-', '-', '-', '-']

  return output_df.sort_values(by=['Numeric', 'Skew', 'Unique'], ascending=False)



def anova(df, feature, label):
  import pandas as pd
  import numpy as np
  from scipy import stats

  groups = df[feature].unique()  
  df_grouped = df.groupby(feature)   
  group_labels = []       
  for g in groups:                   
    g_list = df_grouped.get_group(g)    
    group_labels.append(g_list[label])   

  return stats.f_oneway(*group_labels)



def heteroscedasticity(df, feature, label):
  from statsmodels.stats.diagnostic import het_breuschpagan
  from statsmodels.stats.diagnostic import het_white
  import pandas as pd
  import statsmodels.api as sm
  from statsmodels.formula.api import ols

  # Fit the OLS model: ols(formula='[label]~[feature]', data=df).fit()
  model = ols(formula=(label + '~' + feature), data=df).fit()

  output_df = pd.DataFrame(columns=['LM stat', 'LM p-value', 'F-stat', 'F p-value'])

  try:
    white_test = het_white(model.resid, model.model.exog)
    output_df.loc['White'] = white_test
  except:
    print("Unable to run White test of heteroscedasticity")

  bp_test = het_breuschpagan(model.resid, model.model.exog)
  output_df.loc['Br-Pa'] = bp_test

  return output_df.round(3)



def scatter(feature, label):
  import seaborn as sns
  from scipy import stats
  import matplotlib.pyplot as plt
  import pandas as pd

  # Calculate the regression line
  m, b, r, p, err = stats.linregress(feature, label)

  textstr  = 'y  = ' + str(round(m, 2)) + 'x + ' + str(round(b, 2)) + '\n'
  textstr += 'r2 = ' + str(round(r**2, 2)) + '\n'
  textstr += 'p  = ' + str(round(p, 2)) + '\n'
  textstr += str(feature.name) + ' skew = ' + str(round(feature.skew(), 2)) + '\n'
  textstr += str(label.name) + ' skew = ' + str(round(label.skew(), 2)) + '\n'
  textstr += str(heteroscedasticity(pd.DataFrame(label).join(pd.DataFrame(feature)), feature.name, label.name))

  sns.set(color_codes=True)
  ax = sns.jointplot(feature, label, kind='reg')
  ax.fig.text(1, 0.114, textstr, fontsize=12, transform=plt.gcf().transFigure)
  plt.show()



def bar_chart(df, feature, label):
  import pandas as pd
  from scipy import stats
  from matplotlib import pyplot as plt
  import seaborn as sns

  # Same technique we learned in the bivstats() function to dynamically
  # enter multiple lists of label values for each categorical group
  groups = df[feature].unique()
  df_grouped = df.groupby(feature)
  group_labels = []
  for g in groups:
    g_list = df_grouped.get_group(g)
    group_labels.append(g_list[label])
  
  # Now calculate the ANOVA results
  oneway = stats.f_oneway(*group_labels)

  # Next, calculate t-tests with Bonferroni correction for p-value threshold
  unique_groups = df[feature].unique()
  ttests = []

  for i, group in enumerate(unique_groups):
    for i2, group_2 in enumerate(unique_groups):
      if i2 > i:
        type_1 = df[df[feature] == group]
        type_2 = df[df[feature] == group_2]

        # There must be more than 1 case per group to perform a t-test
        if len(type_1[label]) < 2 or len(type_2[label]) < 2:
          print("'" + group + "' n = " + str(len(type_1)) + "; '" + group_2 + "' n = " + str(len(type_2)) + "; no t-test performed")
        else:
          t, p = stats.ttest_ind(type_1[label], type_2[label])
          ttests.append([group, group_2, t.round(4), p.round(4)])

  if len(ttests) > 0:                # Avoid 'Divide by 0' error
    p_threshold = 0.05 / len(ttests) # Bonferroni-corrected p-value determined
  else:
    p_threshold = 0.05
    
  # Add all descriptive statistics to the diagram
  textstr  = '       ANOVA' + '\n'
  textstr += 'F:            ' + str(oneway[0].round(2)) + '\n'
  textstr += 'p-value:      ' + str(oneway[1].round(2)) + '\n\n'
  textstr += 'Sig. comparisons (Bonferroni-corrected)' + '\n'

  for ttest in ttests:
    if ttest[3] <= p_threshold:
      textstr += ttest[0] + '-' + ttest[1] + ": t=" + str(ttest[2]) + ", p=" + str(ttest[3]) + '\n'

  ax = sns.barplot(df[feature], df[label])
  ax.set_xticklabels(
    ax.get_xticklabels(), 
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large'
  )
  ax.text(1, 0.1, textstr, fontsize=12, transform=plt.gcf().transFigure)
  plt.show()
  


def pearsonchisquare(feature_list, label_list, mincount=5, roundto=4):
  from scipy import stats
  import numpy as np
  import pandas as pd

  # Step 1. Check to see if any value in the contingeny table is less than 5
  flag = False   # Start by creating a variable to indicate (True/False) whether any value is below 5
  crosstab_df = pd.DataFrame(pd.crosstab(feature_list, label_list)) # Generate the contingency table
  for c in crosstab_df:                 # Loop through each column in the contingency table 
    if crosstab_df[c].min() < mincount: # Use the min() funtion to get the lowest value in each column
      flag = True                       # If the min of any column is less than 5, then flag it as 'True'
      break                             # As soon as we find a single value less than 5, no need to keep looping 

  if flag: # If at least one of the values was less than 5, then bucketize
    # Step 2. Determine what the cutoff value is for each of the 10 buckets
    q1 = np.quantile(feature_list, .10)
    q2 = np.quantile(feature_list, .20)
    q3 = np.quantile(feature_list, .30)
    q4 = np.quantile(feature_list, .40)
    q5 = np.quantile(feature_list, .50)
    q6 = np.quantile(feature_list, .60)
    q7 = np.quantile(feature_list, .70)
    q8 = np.quantile(feature_list, .80)
    q9 = np.quantile(feature_list, .90)
  
    # Step 3. Make a new DataFrame to store the 10 quantile values; must be a DataFrame 
    # to work with the crosstab() function even through it only has one column
    bucket_list = pd.DataFrame(columns=['cutoffs']) 
    i = 0 # We need a variable to indicate what index to store the bucket
    # Loop through the raw data and assign a new quantile value in place of the original value
    for v in feature_list: 
      if v < q1:
        bucket_list.loc[i] = q1 
      elif v >= q1 and v < q2:
        bucket_list.loc[i] = q2
      elif v >= q2 and v < q3:
        bucket_list.loc[i] = q3
      elif v >= q3 and v < q4:
        bucket_list.loc[i] = q4
      elif v >= q4 and v < q5:
        bucket_list.loc[i] = q5
      elif v >= q5 and v < q6:
        bucket_list.loc[i] = q6
      elif v >= q6 and v < q7:
        bucket_list.loc[i] = q7
      elif v >= q7 and v < q8:
        bucket_list.loc[i] = q8
      elif v >= q8 and v < q9:
        bucket_list.loc[i] = q9
      else:
        bucket_list.loc[i] = feature_list.max()
      i += 1
    # Step 4. Use the new list (actually a DataFrame of one column) in place of the raw data to create the new conteingency table
    contingency_table = pd.crosstab(bucket_list['cutoffs'], label_list)
  else: # If none of the contingey table values are below 5, then just proceed with the data as is
    contingency_table = pd.crosstab(feature_list, label_list) # Calculate the crosstab

  stat, p, dof, expected = stats.chi2_contingency(contingency_table)
  return [round(stat, roundto), round(p, roundto)]  # I really only want the chi-square stat and the p-value
        
        

def bivstats(df, label):
  from scipy import stats
  import pandas as pd
  import numpy as np

  # Create an empty DataFrame to store output
  output_df = pd.DataFrame(columns=['Stat', '+/-', 'Effect size', 'p-value'])

  for col in df:
    if not col == label:
      if df[col].isnull().sum() == 0:
        if pd.api.types.is_numeric_dtype(df[label]):
          if pd.api.types.is_numeric_dtype(df[col]): # Only calculate r, p-value for the numeric columns
            r, p = stats.pearsonr(df[label], df[col])
            output_df.loc[col] = ['r', np.sign(r), abs(round(r, 3)), round(p, 6)]
            scatter(df[col], df[label])
          else:
            F, p = anova(df[[col, label]], col, label)
            output_df.loc[col] = ['F', '', round(F, 3), round(p, 6)]
            bar_chart(df, col, label)
        else:
          pearsonchisquare(df[col], df[label])
      else:
        output_df.loc[col] = [np.nan, np.nan, np.nan, np.nan]
  

  return output_df.sort_values(by=['Stat', 'Effect size'], ascending=[False, False])
  
  
def correct_skew(label, feature=[]):
  import pandas as pd
  import numpy as np

  transformation = '_skew_corrected'

  def correlate(label, feature_list):
    r_dict = {}
    for f in feature_list:
      r_dict[abs(f.corr(label))] = f
    r_list = list(r_dict.keys())
    r_list.sort(reverse=True)
    return r_dict[r_list[0]]

  def power_tune(feature, start=2, step=0.001, positive=True):
    tuned_list = feature
    i = start
    while tuned_list.skew() < 0:
      if positive:
        tuned_list = feature**(1/i)
      else:
        tuned_list = feature**i
      i+=0.001
    return i

  if len(feature) < 1:
    feature = label

  if feature.skew() > 0:
    sq = np.sqrt(feature)
    cb = np.cbrt(feature)
    if feature.min() >= 0:
      log = np.log(feature + 1)
    else:
      log = feature**(1 / 3.5)
    if sq.skew() > 0 and cb.skew() < 0:
      i = power_tune(feature)
      tuned = feature**i
    else:
      tuned = feature**(1 / 2.5)
  else:
    sq = feature**2
    cb = feature**3
    log = np.exp(feature)
    if sq.skew() < 0 and cb.skew() > 0:
      i = power_tune(feature, positive=False)
      tuned = feature**i
    else:
      tuned = feature**2.5

  if len(feature) < 1:
    return [transformation, correlate(label, [feature, sq, sq, log, tuned])]
  else:
    return [transformation, correlate(label, [feature, sq, sq, log])]
  
  
  