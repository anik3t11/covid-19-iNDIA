#!/usr/bin/env python
# coding: utf-8

# # Exploratory Analysis of Covid-19 in India using Machine Learning

# ##  By Aniket Vyas En18El301019

# ![](https://res.cloudinary.com/people-matters/image/upload/fl_immutable_cache,w_624,h_351,q_auto,f_auto/v1585158318/1585158316.jpg)

# # Introduction to Covid-19 in India
# A novel Corona virus is a category of pathogens, which mostly attacks on the respiratory system of human beings. Corona virus outbreaks are emerging earlier also in the form of Severe Acute Respiratory Syndrome (SARS) and Middle East Respiratory Syndrome (MERS).Now, in the present times, it emerges as a COVID-19, which is caused by the SARS2 Coronavirus and poses a significant risk to the human race. 
# 
# In December 2019, a number of patients with pneumonia symptoms were reported in Wuhan and Hubei Province of China and later identified as symptoms caused due to the spread of corona virus. Epidemiologically, these patients were later on found to be linked to an animal and seafood market of Wuhan. Later on, the Wuhan city of China was recognized as the epicentre of the COVID-19 disease and claimed for spreading the disease across the world. Around 41 lab confirmed COVID-19 patients reported and admitted to hospital up to January 2, 2020 in China. These patients have symptoms of coughing, sneezing, breathing problems, chest pain, indigestion, and respiratory illness. It was also observed that most of these patients are already suffering from varied diseases such as hypertension, diabetes and cardiovascular.   
# 
# According to China National Health Commission, 17 deaths reported in China with COVID-19 up to January 22, 2020 and within four days death rate raised to triple with 5502 confirmed cases. By the end of January 2020, 7734 confirmed cases have been reported in China along with 90 cases in other countries such as Thailand, Japan, Malaysia, Iran, Italy, India, USA, Canada, Taiwan, Vietnam, France, Nepal, Cambodia, Germany, Singapore, Korea, United Arab Emirates, Sri Lanka, The Philippines, Australia and Finland. 
# 
# Furthermore, WHO on 30 Jan declares Public health emergency of international concern due to the severity of the disease. The International Committee on Taxonomy of Viruses named this corona virus as a Severe Acute Respiratory Syndrome Coronavirus-2 (SARS-CoV-2) and the disease caused by the virus designated as Coronavirus Disease-19 (COVID-19) by WHO. 

# # How it started in India?
# 
# ##### The first *COVID-19* case was reported on 30th January 2020 when a student arrived *Kerala* from **Wuhan**. Just in next 2 days, Kerela reported 2 more cases. For almost a month, no new cases were reported in India, however, on 2nd March 2020, five new cases of corona virus were reported in Kerala again and since then the cases have been rising affecting *25* states.

# ### Corona Virus Explained in Simple Terms:
# - Let's say Raj got infected yesterday, but he won't know it untill next 14 days
# - Raj thinks he is healthy but he is infecting 10 persons per day
# - Now these 10 persons think they are completely healthy, they travel, go out and infect 100 others
# - These 100 persons think they are healthy but they have already infected 1000 persons
# - No one knows who is healthy or who can infect you
# - All you can do is be responsible, stay in quarentine
# 
# 

# ![](https://i.imgur.com/nf2kMhF.jpg)

# #### Importing Dataset  

# In[14]:


import pandas as pd
import numpy as np


# In[15]:


case_df = pd.read_csv('covid_19_india.csv')
case_df


# In[16]:


vac_df = pd.read_csv('covid_vaccine_statewise.csv')
vac_df


# In[ ]:





# In[ ]:





# ## Data Prepration and Cleaning
# - Load the data set into a dataframe using Pandas
# - Explore the number of rows & columns ,ranges of values,etc.
# - Handle missing,incorrectand invalid data

# In[17]:


case_df.info()


# In[18]:


case_df.describe()


# In[19]:


#Drop Lat & Long
case_df.drop(['ConfirmedIndianNational', 'ConfirmedForeignNational','Sno','Time'], inplace=True, axis=1)

#Rename Cured to Recovered
case_df.rename(columns = {'Cured':'Recovered'}, inplace = True) 
#Rename State/UnionTerritory to Province_State
case_df.rename(columns = {'State/UnionTerritory':'Province_State'}, inplace = True) 


# In[20]:


case_df.info()


# In[21]:


case_df


# In[22]:


#Add New Cases
case_df['Prev_Confirmed'] = case_df.groupby('Province_State')['Confirmed'].shift(1)
case_df['New Cases'] = case_df['Confirmed'] - case_df['Prev_Confirmed']
case_df.drop('Prev_Confirmed',inplace = True,axis=1)

#Add New Recovered Cases
case_df['Prev_Recovered'] = case_df.groupby('Province_State')['Recovered'].shift(1)
case_df['New Recovered'] = case_df['Recovered'] - case_df['Prev_Recovered']
case_df.drop('Prev_Recovered',inplace = True,axis=1)

#Add New Deaths Cases
case_df['Prev_Deaths'] = case_df.groupby('Province_State')['Deaths'].shift(1)
case_df['New Deaths'] = case_df['Deaths'] - case_df['Prev_Deaths']
case_df.drop('Prev_Deaths',inplace = True,axis=1)


# In[23]:


case_df.isnull().sum()


# In[24]:


#few Null data So, Remove it
case_df['New Cases'].fillna(0, inplace=True)
case_df['New Recovered'].fillna(0, inplace=True)
case_df['New Deaths'].fillna(0, inplace=True)


# In[25]:


#type_check
case_df['New Deaths'].dtypes


# In[26]:


#type change to integer
case_df['New Deaths'] = case_df['New Deaths'].astype(int)
case_df['New Cases'] = case_df['New Cases'].astype(int)
case_df['New Recovered'] =case_df['New Recovered'].astype(int)


# In[27]:


case_df['Province_State'].unique()


# In[28]:


#Remove non sate data
#drop 'Cases being resigned to states' & 'unassigned'
df_dump = case_df['Province_State'] == 'Unassigned'
case_df.drop(case_df[df_dump].index,inplace=True)
df_dump = case_df['Province_State'] == 'Cases being reassigned to states'
case_df.drop(case_df[df_dump].index,inplace=True)


# In[29]:


def change_state_name(state):
    if state =='Odisha':
        return 'Orissa'
    elif state =='Telenagana':
        return 'Telangana'
    return state

case_df['Province_State'] = case_df.apply(lambda x: change_state_name(x['Province_State']),axis=1)
    


# In[30]:


last_date = case_df.Date.max()
state_cases = case_df.copy()
state_cases = state_cases[state_cases['Date']==last_date]
state_cases.drop(['Date','New Cases','New Recovered','New Deaths'],inplace = True,axis=1)


# In[31]:


#Add active ,Deaths/Recovered,Mortality & Recovered Rate(per 100)
state_cases['Active']= state_cases['Confirmed']-(state_cases['Deaths'] +state_cases['Recovered'])
state_cases['Active'] = state_cases['Active'].astype(int)
state_cases['Mortality Rate(per 100)'] = np.round(100*state_cases["Deaths"]/state_cases["Confirmed"],2)
state_cases['Recovered Rate(per 100)'] = np.round(100*state_cases['Recovered']/state_cases['Confirmed'],2)


# In[32]:


state_cases.reset_index(drop=True,inplace=True)
state_cases.head()


# In[33]:


#Day wise copy data
daywise_df = case_df.groupby('Date').sum().reset_index()
daywise_df.head()


# In[34]:


#check Null
daywise_df.isnull().sum()


# In[35]:


#Add death/Recovered ,Mortality & Recovered Rate(per100)
case_df['Mortality Rate(per 100)'] = np.round(100*case_df['Deaths']/case_df['Confirmed'],2)
case_df['Recovered Rate(per 100)'] = np.round(100*case_df['Recovered']/case_df['Confirmed'],2)


# In[36]:


#check null
case_df.isnull().sum()


# In[37]:


#few null data so remove it
case_df['Mortality Rate(per 100)'] = case_df['Mortality Rate(per 100)'].replace(np.nan,0)
case_df['Recovered Rate(per 100)'] = case_df['Recovered Rate(per 100)'].replace(np.nan,0)


# In[38]:


#check null
case_df.isnull().sum()


# ## Exploratory Analysis and Visualization
# 
# - Matplotlib - for visualization (Plotting Graphs)
# - Seaborn - for visualization (Plotting colorful graphs)
# - Plotly - for interactive plots
# - Explore distributions of numeric columns using histograms etc.
# - Explore relationship between columns using scatter plots, bar charts etc.

# Install plotly -express

# In[39]:


get_ipython().system('pip install plotly-express')


# In[40]:


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import plotly_express as px
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] =14
matplotlib.rcParams['figure.figsize']=(9,5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# ## Total Reported Cases

# In[41]:


total_cases = state_cases.copy()
total_cases.drop(['Province_State','Mortality Rate(per 100)','Recovered Rate(per 100)'],inplace = True,axis=1)
total = total_cases.sum()
total.name = "Total"
df_t = pd.DataFrame(total,dtype=float).transpose()
df_t["Mortality Rate(per 100)"] = np.round(100*df_t["Deaths"]/df_t["Confirmed"],2)
df_t.style.background_gradient(cmap='Purples',axis=1).format("{:.2f}").format("{:.0f}",subset=["Confirmed","Deaths","Recovered","Active"])


# In[42]:


india_confirmed = total.Confirmed
india_active = total.Active
india_recovered = total.Recovered
india_deaths = total.Deaths
labels = ['Active','Recovered','Deaths']
sizes = [india_active,india_recovered,india_deaths]
color= ['#66b3ff','green','red']
explode = []

for i in labels:
    explode.append(0.02)

plt.figure(figsize= (15,5))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=9, explode =explode,colors = color)
centre_circle = plt.Circle((0,0),0.30,fc='white')

fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('India COVID-19 Cases',fontsize = 20)
plt.axis('equal')  
plt.tight_layout()


# #### A total of 32036511 COVID-19 confirmed cases have been reported in India till 11 August 2021 with 386351 active cases (1.2%),31220981 cured/discharged (97.5%),and 429179 deaths (1.3%).

# ## Statewise Insights

# In[43]:


state_case_df = state_cases.copy()
state_case_df = state_case_df.set_index('Province_State')
state_case_df.sort_values('Confirmed',ascending = False).fillna(0).style.background_gradient(cmap='Blues',subset=['Confirmed'])                        .background_gradient(cmap='Reds',subset=["Deaths"])                        .background_gradient(cmap='Greens',subset=["Recovered"])                        .background_gradient(cmap='Purples',subset=["Active"])                        .background_gradient(cmap='YlOrBr',subset=["Mortality Rate(per 100)"])                        .background_gradient(cmap='Greens_r',subset=["Recovered Rate(per 100)"])


# Maharashtra state has most number of covid 19  case with about 6363442 case in total

# ### India Covid19 Cases growth (Day Wise)

# In[44]:


daywise_df


# In[45]:


fig, ax = plt.subplots(figsize=(15, 10))
ax.clear()
#fig = sns.lineplot(x="Date",y="Active",data = covid_india_dayswise_df ,color="y",label="Active")
fig = sns.lineplot(x="Date",y="Recovered",data = daywise_df ,color="green",label="Recovered")
fig = sns.lineplot(x="Date",y="Deaths",data = daywise_df ,color="r",label="Deaths")
fig = sns.lineplot(x="Date",y="Confirmed",data = daywise_df ,color="c",label="Confirmed")
fig.set_xlabel('Date\n',size=15,color='#4bb4f2') 
fig.set_ylabel('Number of Cases\n',size=15,color='#4bb4f2') 
fig.set_title('India Covid-19 Cases',size=25,color='navy')
fig.ticklabel_format(style='plain', axis='y',useOffset=False)


# In[46]:


df1 = daywise_df.melt(id_vars='Date', value_vars=['New Cases','New Deaths','New Recovered'], 
                 var_name='Cases', value_name='Cases Count')
fig = px.line(df1, x="Date", y="Cases Count",color='Cases')
fig.update_layout(title="India Covid-19 Daily Cases", xaxis_title="", yaxis_title="")
fig.show()


# ### From above line plots:
#  - As we can see in the graph there is peak in september 2020 and sharp peak on May 7 2021. 
# - According to graph there is chances of getting next wave in next month but as now as there are availability of vaccines and medicines which can easily defeat covid virus so there are very less chances of getting next wave soon.

# # Covid19 State Wise 

# In[47]:


fig = px.line(case_df,x='Date', y='Confirmed', color='Province_State',title='India growth COVID19 Cases ')
fig.show()


# ## From the above line plots for the states:
# 
# - The cases and deaths increase monotincally
# - Maharashtra and kerala has shown the greatest rise in the number of covid Cases. On the other hand karnataka has 3 highest number of confirmed covid case.

# ## Top 10 States 

# #### Bar Plot Analysis

# In[48]:


def plot_hbar(df, col, n, hover_data=[]):
    fig = px.bar(df.sort_values(col).tail(n), 
                 x=col, y="Province_State", color='Province_State',  
                 text=col, orientation='h', width=700, hover_data=hover_data,
                 color_discrete_sequence = px.colors.qualitative.Dark24)
    fig.update_layout(title=col, xaxis_title="", yaxis_title="", 
                      yaxis_categoryorder = 'total ascending',
                      uniformtext_minsize=8, uniformtext_mode='hide')
    fig.show()


# In[49]:


plot_hbar(state_cases,'Confirmed',10)


# ### From the above bar for the states
# 
# - Maharashtra & Kerala has the highest number of confirmed case. with maharshtra has about twice the number of confirmed case as compare to kerala with almost 30k confirmed case. Karnataka has 3 highest number of confirmed case.

# In[50]:


plot_hbar(state_cases, 'Deaths', 10)


# ###  From the above plot we can conclude
# 
# - Maharastra and karntaka being top 2 case with highest number of death.
# - Reason being probably both state have highest number of migrants from other state for work.
# - While tamilnadu stands on 3 place in terms of death case.

# In[51]:


plot_hbar(state_cases, 'Recovered', 10)


# ### FRom the above bar plot we can conclude
# - Maharashtra and kerala has shown greatest rise in the number of recovered cases reason may be as both the state have highets number of confiremd cases too.
# - Karnataka stands at 3 position in term of covid recovery position.

# In[52]:


plot_hbar(state_cases, 'Active', 10)


# ### From the above plot we can conclude that
# - Kerala has highest number of active cases dated 11 August 2021 while maharshtra stands on 2 place in terms of active case. Kerala has almost thrice the number of active cases as compare to maharshtra which stands on 2 place in terms of active cases

# In[53]:


plot_hbar(state_cases, 'Mortality Rate(per 100)', 10)


# ### From the above plot
# 
# - As we all know how the covid affect the life of people from tht above plot we can conclude tha

# In[54]:


plot_hbar(state_cases, 'Recovered Rate(per 100)', 10)


# In[55]:


fig = px.scatter(state_cases.sort_values('Deaths', ascending=False).iloc[:20, :], 
                 x='Confirmed', y='Deaths', color='Province_State', size='Confirmed', 
                 height=700, text='Province_State', log_x=True, log_y=True, 
                 title='Confirmed vs Deaths (Scale in log10)')
fig.update_traces(textposition='top center')
fig.update_layout(showlegend=False)
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()


# ## Extracting Important Insights from the Project

# ###  Q What is the Total number of COVID-19 cases in India?

# In[56]:


total_cases = state_cases.copy()
total_cases.drop(['Province_State','Mortality Rate(per 100)','Recovered Rate(per 100)'],inplace = True,axis=1)
total = total_cases.sum()
total.name = "Total"
df_t = pd.DataFrame(total,dtype=float).transpose()
df_t["Mortality Rate(per 100)"] = np.round(100*df_t["Deaths"]/df_t["Confirmed"],2)
df_t.style.background_gradient(cmap='Purples',axis=1).format("{:.2f}").format("{:.0f}",subset=["Confirmed","Deaths","Recovered","Active"])


# A total of 32036511 COVID-19 confirmed cases have been reported in Indian as from 11 August 2021 with 386351 active cases (1.2%),31220981 cured/discharged (97.45%),and 429179 deaths (1.3%).

# ###  Q Which state is the most recent with COVID-19 Deaths case?

# In[57]:


top_state = state_cases.sort_values('Deaths', ascending= False).head(5)
#top_state
top_state.reset_index(drop=True,inplace=True)
top_state.style.background_gradient(cmap='Blues',subset=["Confirmed"])                        .background_gradient(cmap='Reds',subset=["Deaths"])                        .background_gradient(cmap='Greens',subset=["Recovered"])                        .background_gradient(cmap='Purples',subset=["Active"])                        .background_gradient(cmap='YlOrBr',subset=["Mortality Rate(per 100)"])                        .background_gradient(cmap='Greens_r',subset=["Recovered Rate(per 100)"])


# Maharashtra state is the most recent with COVID-19 Deaths cases 134201.

# ### Q Which state is the High Mortality Rate(per 100) in COVID-19?

# In[58]:


top_state = state_cases.sort_values('Mortality Rate(per 100)', ascending= False).head(5)
#top_state
top_state.reset_index(drop=True,inplace=True)
top_state.style.background_gradient(cmap='Blues',subset=["Confirmed"])                        .background_gradient(cmap='Reds',subset=["Deaths"])                        .background_gradient(cmap='Greens',subset=["Recovered"])                        .background_gradient(cmap='Purples',subset=["Active"])                        .background_gradient(cmap='YlOrBr',subset=["Mortality Rate(per 100)"])                        .background_gradient(cmap='Greens_r',subset=["Recovered Rate(per 100)"])


# Punjab state is with Highest mortality Rate (per 100) 2.72 in COVID-19

# ### Q Which state is the High Recovered Rate(per 100) in COVID-19?   

# In[59]:


top_state = state_cases.sort_values('Recovered Rate(per 100)', ascending= False).head(5)
#top_state
top_state.reset_index(drop=True,inplace=True)
top_state.style.background_gradient(cmap='Blues',subset=["Confirmed"])                        .background_gradient(cmap='Reds',subset=["Deaths"])                        .background_gradient(cmap='Greens',subset=["Recovered"])                        .background_gradient(cmap='Purples',subset=["Active"])                        .background_gradient(cmap='YlOrBr',subset=["Mortality Rate(per 100)"])                        .background_gradient(cmap='Greens_r',subset=["Recovered Rate(per 100)"])


# Dadar and Nagar Haveli and daman and diu state is the highest recovered reate (per 100) 99.92 in COVID-19

# ###  Q Find the dail Covid-19 Mortality Rates (per 100) in India

# In[60]:


daywise_df["Mortality Rate(per 100)"] = np.round(100*daywise_df["Deaths"]/daywise_df["Confirmed"],2)
daywise_df[["Date","Mortality Rate(per 100)"]]


# In[61]:


fig = px.line(daywise_df, x="Date", y="Mortality Rate(per 100)", color_discrete_sequence=['lightcoral'],height=600)
fig.update_layout(title="India Covid-19 Mortality Rate (per 100)", xaxis_title="", yaxis_title="")
fig.show()


# ### Q Find the daily Covid-19 Recovered Rate (per 100) in India 

# In[62]:


daywise_df["Recovered Rate(per 100)"] = np.round(100*daywise_df["Recovered"]/daywise_df["Confirmed"],2)
daywise_df[["Date","Recovered Rate(per 100)"]]


# In[63]:


fig = px.line(daywise_df, x="Date", y="Recovered Rate(per 100)", color_discrete_sequence=['darkseagreen'],height=600)
fig.update_layout(title="India Covid-19 Recovered Rate(per 100)", xaxis_title="", yaxis_title="")
fig.show()


# ###  Find the Daily Covid-19 Incidence Rate(per 100) in India

# In[64]:


daywise_df["Incidence Rate(per 100)"] = np.round(100*daywise_df["New Cases"]/daywise_df["Confirmed"],2)
daywise_df[["Date","Incidence Rate(per 100)"]]


# In[65]:


fig = px.line(daywise_df, x="Date", y="Incidence Rate(per 100)", color_discrete_sequence=['red'],height=600)
fig.update_layout(title="India Covid-19 Incidence Rate(per 100)", xaxis_title="", yaxis_title="")
fig.show()


# In[ ]:




