import streamlit as st
import pandas as pd
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import io
import base64 
import time
from prophet import Prophet
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler

def set_page_config():
    PAGE_CONFIG = {
        "page_title": "Data Observality",
        "page_icon":"",
        "layout": "wide",
        "initial_sidebar_state": "expanded",
    }
    st.set_page_config(**PAGE_CONFIG)

set_page_config()

global selected_date_col


def uploadedFiles():
    
    st.sidebar.markdown("""
            <style>
            .sidebar .sidebar-content {
                background-color: #EEF4FE; /* Yellow background */
                color: black; /* Text color for contrast */
            }
            .sidebar .sidebar-header {
                text-align: center; /* Center-align the header */
                color: black; /* Text color for contrast */
                font-size: 24px; /* Increase font size */
                padding: 10px 0; /* Add some padding */
            }
            </style>
        """, unsafe_allow_html=True)

    st.sidebar.title("Data Observation Dashboard")
    yesterday_file=''
    file_type = source = st.sidebar.selectbox("Data Source", options=('Excel', 'CSV', 'Snowflake'), index=None,placeholder="Select a Source")
    if file_type in ["Excel", "CSV"]:
        yesterday_file = st.sidebar.file_uploader("Upload yesterday's file", type=["csv", "xlsx"])
    today_file = st.sidebar.file_uploader("Upload today's file", type=["csv", "xlsx"])
    return file_type, yesterday_file, today_file

@st.cache_data
def compare_data(data1, data2, date_col):

    data1=data1.copy()
    data2=data2.copy()

    data1,data2_filtered=filter_data_by_date_range(data1, data2, date_col)

    # Ensure the date column is in datetime format
    data1[date_col] = pd.to_datetime(data1[date_col]).dt.date
    data2_filtered[date_col] = pd.to_datetime(data2_filtered[date_col]).dt.date

    # Total rows in each file
    total_rows_file1 = len(data1)
    total_rows_file2 = len(data2)
    print("Inside Compare Data",total_rows_file2)

    categorical_columns = data2_filtered.select_dtypes(include=['object']).columns.to_list()
    categorical_columns.append(date_col)
    
    new_records = data2_filtered[~data2_filtered[categorical_columns].isin(data1[categorical_columns].to_dict(orient='list')).all(axis=1)]
    
    deleted_records=data1[~data1[categorical_columns].isin(data2_filtered[categorical_columns].to_dict(orient='list')).all(axis=1)]
    
    # Identify rows that are completely missing in file 2 compared to file 1
    Changes_in_recent_File=data2_filtered[~data2_filtered.apply(tuple, 1).isin(data1.apply(tuple, 1))]
    Changes_in_old_File = data1[~data1.apply(tuple, 1).isin(data2_filtered.apply(tuple, 1))]


    # Column comparison
    yesterday_columns = set(data1.columns)
    today_columns = set(data2.columns)
    column_comparison = {
        'all_columns_present': today_columns.issubset(yesterday_columns),
        'missing_columns': list(today_columns - yesterday_columns) if not today_columns.issubset(
            yesterday_columns) else []
    }

    return {
        'total_rows_file1': total_rows_file1,
        'total_rows_file2': total_rows_file2,
        'Changes_in_recent_File': Changes_in_recent_File,
        'Changes_in_old_File': Changes_in_old_File,
        'new_records':new_records,
        'deleted_records':deleted_records
    }



# Function to display comparison results
@st.cache_data
def filter_data_by_date_range(df1, df2, date_column):
    df1=df1.copy()
    df2=df2.copy()
    # Find the overlapping date range
    max_start_date = max(df1[date_column].min(), df2[date_column].min())
    min_end_date = min(df1[date_column].max(), df2[date_column].max())

    # Filter both DataFrames to the overlapping date range
    df1_filtered = df1[(df1[date_column] >= max_start_date) & (df1[date_column] <= min_end_date)]
    df2_filtered = df2[(df2[date_column] >= max_start_date) & (df2[date_column] <= min_end_date)]

    return df1_filtered, df2_filtered


@st.cache_data
def display_comparison_results(results, data1, data2,Target_col, file1_name, file2_name):
    data1 = data1.copy()
    data2 = data2.copy()
    data1,data2=filter_data_by_date_range(data1, data2, 'Date') 
    
    # Convert the 'Date' column to datetime if it's not already
    data2['Date'] = pd.to_datetime(data2['Date']).dt.date
    data1['Date'] = pd.to_datetime(data1['Date']).dt.date

    # Get the maximum date from data1 (yesterday's data)
    yesterdayMaxDate = data1['Date'].max()

    # Count rows in data2 (today's data) that are greater than or less than/equal to yesterday's max date
    todayRowsGreater = len(data2[data2['Date'] > yesterdayMaxDate])
    todayRowsLessEqual = len(data2[data2['Date'] <= yesterdayMaxDate])

    # Find new rows in data2 that are not in data1
    new_rows = data2.merge(data1, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
    new_rows_count = new_rows.shape[0]

    # Identify new brands in data2 that are not present in data1
    Delta_Brand = len(set(data2['Brnd_Name'].unique()) - set(data1['Brnd_Name'].unique()))
    #Delta_Indicator = set(data2['Prscrbr_Type'].unique()) - set(data1['Prscrbr_Type'].unique())
    categorical_columns = data1.select_dtypes(include=['object']).columns.to_list()
    
    if 'Date' in categorical_columns:
        categorical_columns.remove('Date')
        
    print(" Inside display_comparison_results",categorical_columns)
    # Initialize a list to hold new values
    new_values = []

    for col in categorical_columns:
            new_in_data2 = set(data2[col].astype(str).unique()) - set(data1[col].astype(str).unique())
            if new_in_data2:
                new_values.append(f"In column '{col}', new values: {', '.join(new_in_data2)}")

        # Display results in Streamlit if there are new values
    if new_values:
        #st.markdown("### New categorical values found:")
        for item in new_values:
            st.markdown(f"<span style='color:black;'>- {item}</span>", unsafe_allow_html=True)
    new_values=list(set(new_values))
    
    del_values=[]
    
    for col in categorical_columns:
            del_in_data2 = set(data1[col].astype(str).unique()) - set(data2[col].astype(str).unique())
            if del_in_data2:
                del_values.append(f"In column '{col}', deleted values: {', '.join(new_in_data2)}")

        # Display results in Streamlit if there are new values
    print(" Inside display_comparison_results ",del_in_data2)    
    if del_values:
        #st.markdown("### New categorical values found:")
        for item in del_values:
            st.markdown(f"<span style='color:black;'>- {item}</span>", unsafe_allow_html=True)
    
    # Target Value Increase 
    categorical_columns.append('Date')
    grouped_data1 = data1.groupby(categorical_columns)[Target_col].sum().reset_index()
    grouped_data2 = data2.groupby(categorical_columns)[Target_col].sum().reset_index()

    # Merge data1 and data2 on the categorical columns
    merged_data = pd.merge(grouped_data1, grouped_data2, on=categorical_columns, how='outer', suffixes=('_data1', '_data2'))

    # Fill NaN values with 0 (assuming missing categories should be counted as zero)
    merged_data.fillna(0, inplace=True)

    # Calculate differences
    merged_data['Difference'] = merged_data[Target_col + '_data2'] - merged_data[Target_col + '_data1']

    # Find records where there are changes
    changes = merged_data[merged_data['Difference'] != 0]

    # Count the number of changes
    num_changes = changes.shape[0]

    # Display results
    st.markdown(f"<span style='color:black;'>- Number of Rows for which the {Target_col} increases: {num_changes}</span>", unsafe_allow_html=True)
    print(changes)
    
    if num_changes > 0:
        st.dataframe(changes)    
        
    
    
    



@st.cache_resource           
def timePlusBoosting(trainingdata,testingdata,target_col):
    trainingdata=trainingdata.copy()
    testingdata=testingdata.copy()
    
    #filter the columns which ever is required
    trainingdata = trainingdata[['Date','Brnd_Name','Tot_Clms','Tot_Day_Suply','Tot_Drug_Cst','Tot_30day_Fills']]
    testingdata = testingdata[['Date','Brnd_Name','Tot_Clms','Tot_Day_Suply','Tot_Drug_Cst','Tot_30day_Fills']]

    # One-hot encoding
    trainingdata_encoded = pd.get_dummies(trainingdata, columns=['Brnd_Name'])
    testingdata_encoded = pd.get_dummies(testingdata, columns=['Brnd_Name'])

    testingdata_encoded = testingdata_encoded.reindex(columns=trainingdata_encoded.columns, fill_value=0)

    scaler = MinMaxScaler()

    trainingdata_encoded[['Tot_Clms', 'Tot_Day_Suply', 'Tot_Drug_Cst']] = scaler.fit_transform(trainingdata_encoded[['Tot_Clms', 'Tot_Day_Suply', 'Tot_Drug_Cst']])
    testingdata_encoded[['Tot_Clms', 'Tot_Day_Suply', 'Tot_Drug_Cst']] = scaler.fit_transform(testingdata_encoded[['Tot_Clms', 'Tot_Day_Suply', 'Tot_Drug_Cst']])

    X_train = trainingdata_encoded.drop(['Date', 'Tot_30day_Fills'], axis=1)
    y_train = trainingdata_encoded[target_col]


    X_test = testingdata_encoded.drop(['Date', 'Tot_30day_Fills'], axis=1)
    y_test = testingdata_encoded[target_col]


    gbr = GradientBoostingRegressor(random_state=42)
    gbr.fit(X_train, y_train)


    y_pred = gbr.predict(X_test)
    testingdata['predicted']=y_pred
    residuals = np.abs(y_test - y_pred)
    testingdata['residuals']=residuals

    #Threshold
    threshold_data = []

    for brand in testingdata['Brnd_Name'].unique():
        brand_df = testingdata[testingdata['Brnd_Name'] == brand]
        brand_df = brand_df.reset_index()
        mean_residuals = np.mean(brand_df['residuals'])
        std_residuals = np.std(brand_df['residuals'])

        # Threshold: mean + 2 * std deviation
        threshold = mean_residuals + (2) * std_residuals
        
        # Append the brand and its threshold to the list
        threshold_data.append({'Brnd_Name': brand, 'threshold': threshold})

        # Mark anomalies
        testingdata.loc[testingdata['Brnd_Name'] == brand, 'pred'] = (testingdata['residuals'] > threshold).astype(int)

    threshold_df = pd.DataFrame(threshold_data)

    #all anomalies in this variable
    anomaliesGetStd = testingdata[testingdata['pred']==True].reset_index()
    anomaliesGetStd.drop(['index'],axis=1,inplace=True)
            
    return testingdata
 
def get_datetime_columns(df):
    # Get columns with datetime format
    datetime_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    return datetime_columns

@st.cache_data
def brand_Comparison_Over_Time(trainingdata, testingdata, Target_Col):
    trainingDataBrandGraph = trainingdata.copy()

    trainingDataBrandGraph['Date'] = pd.to_datetime(trainingDataBrandGraph['Date'], errors='coerce')

    # Determine the appropriate time grouping based on the range of dates
    if trainingDataBrandGraph['Date'].dt.year.nunique() > 1:
        trainingDataBrandGraph['time_period'] = trainingDataBrandGraph['Date'].dt.year
    elif trainingDataBrandGraph['Date'].dt.to_period('M').nunique() > 1:
        trainingDataBrandGraph['time_period'] = trainingDataBrandGraph['Date'].dt.to_period('M')
    else:
        trainingDataBrandGraph['time_period'] = trainingDataBrandGraph['Date'].dt.to_period('D')

    training_grouped = trainingDataBrandGraph.groupby(['time_period', 'Brnd_Name'])[Target_Col].sum().reset_index()

    # Testing Data Processing
    testingdata=testingdata.copy()
    testingdata['Date'] = pd.to_datetime(testingdata['Date'])

    if testingdata['Date'].dt.year.nunique() > 1:
        testingdata['time_period'] = testingdata['Date'].dt.year
    elif testingdata['Date'].dt.to_period('M').nunique() > 1:
        testingdata['time_period'] = testingdata['Date'].dt.to_period('M')
    else:
        testingdata['time_period'] = testingdata['Date'].dt.to_period('D')

    testing_grouped = testingdata.groupby(['time_period', 'Brnd_Name'])[Target_Col].sum().reset_index()

    # Merge the training and testing data on time_period and brand
    merged_data = pd.merge(training_grouped, testing_grouped, on=['time_period', 'Brnd_Name'], suffixes=('_train', '_test'),how='left')

    # Calculate percentage change
    merged_data['pct_change'] = ((merged_data[f'{Target_Col}_test'] - merged_data[f'{Target_Col}_train']) / merged_data[f'{Target_Col}_train']) * 100

    # Detect significant changes
    threshold = 20
    significant_changes = merged_data[merged_data['pct_change'].abs() > threshold]

    # Set the figure size explicitly
    fig2, ax2 = plt.subplots(figsize=(8, 4))  # Adjusted to fit the div height
    for brand in merged_data['Brnd_Name'].unique():
        brand_data = merged_data[merged_data['Brnd_Name'] == brand]
        ax2.plot(brand_data['time_period'].astype(str), brand_data[f'{Target_Col}_test'], marker='o', markersize=5, label=f'Current - {brand}')
        ax2.plot(brand_data['time_period'].astype(str), brand_data[f'{Target_Col}_train'], marker='x', markersize=5, linestyle='--', label=f'Previous - {brand}')

    # Place the legend inside the plot
    plt.legend(fontsize=6, loc='center left', bbox_to_anchor=(1,0.8))

    # Set axis labels and title
   # ax2.set_xlabel('Time Period', fontsize=10)
    #ax2.set_ylabel('Total Value', fontsize=10)
    #ax2.set_title('Comparison of Brand Level Distribution Over Time', fontsize=12)

    # Adjust layout to fit everything nicely
    plt.tight_layout()

    return fig2
    
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')  # Ensure entire figure is saved
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return img_base64

@st.cache_data
def BarChartNumberAnomaliesDetected(testingData):
    fig, ax = plt.subplots(figsize=(6, 3))  # Increased height to 3

    goodRecords = len(testingData[testingData['pred'] == 0])
    anomalousRecords = len(testingData[testingData['pred'] == 1])

    # Bar plots
    ax.barh(np.arange(1), goodRecords, color='#D8BFD8', label='Non-Anomalous Records')
    ax.barh(np.arange(1), anomalousRecords, color='#4B0082', label='Anomalous Records', left=goodRecords)

    ax.set_yticks(np.arange(1))
    ax.set_yticklabels([''])

    #ax.set_title('Non-Anomalous Data vs Anomalous Data', pad=17,fontsize=10.2)  # Added pad to increase distance from the plot
    plt.xticks([])

    # Annotate bars with values
    ax.text(goodRecords / 2, 0, str(goodRecords), va='center', ha='center', color='black', fontsize=20)
    ax.text(goodRecords + anomalousRecords / 2, 0, str(anomalousRecords), va='center', ha='center', color='black',
            fontsize=20)

    plt.tight_layout()

    # Move legend
    #plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), bbox_transform=ax.transAxes, fontsize=8)
    return fig
@st.cache_data
def Data_Profile(df, filter,Target_Col):
    # Create the initial group_df with mean, std, count, min, max
    group_df = df.groupby([filter])[Target_Col].agg(['mean', 'std', 'count', 'min', 'max']).reset_index()

    # Separate the data into normal and anomalous based on 'pred' column
    normal_df = df[df['pred'] == 0]
    anomalous_df = df[df['pred'] == 1]

    # Get normal and anomalous ranges
    normal_ranges = normal_df.groupby([filter])[Target_Col].agg(['min', 'max']).reset_index()
    anomalous_ranges = anomalous_df.groupby([filter])[Target_Col].agg(['min', 'max']).reset_index()

    # Rename the columns for clarity
    normal_ranges.rename(columns={'min': 'Normal_Min', 'max': 'Normal_Max'}, inplace=True)
    anomalous_ranges.rename(columns={'min': 'Anomalous_Min', 'max': 'Anomalous_Max'}, inplace=True)

    # Merge the ranges with the main group_df
    group_df = pd.merge(group_df, normal_ranges, on=filter, how='left')
    group_df = pd.merge(group_df, anomalous_ranges, on=filter, how='left')

    return group_df    

@st.cache_data
def filter_data_by_period(data, date_column, period):
    # Ensure the date_column is in datetime format
    data[date_column] = pd.to_datetime(data[date_column], errors='coerce')

    reference_date = data[date_column].max()
    if period == 'R6M':
        cutoff_date = reference_date - relativedelta(months=6)
    elif period == 'R12M':
        cutoff_date = reference_date - relativedelta(months=12)
    elif period == 'R18M':
        cutoff_date = reference_date - relativedelta(months=18)
    elif period == 'R24M':
        cutoff_date = reference_date - relativedelta(months=24)
    else:
        raise ValueError("Unsupported period. Use 'R6M', 'R12M', 'R18M', or 'R24M'.")

    print("Cutoff Date:", cutoff_date)
    print("Reference Date:", reference_date)

    return data[data[date_column] >= cutoff_date]

@st.cache_data
def Anomalies_Brand(testingData,filter_Col):
    # Filter and count anomalous records
    anomalousData = testingData[testingData['pred'] == 1].groupby([filter_Col])['pred'].count().reset_index()
    anomalousData.rename(columns={'pred': 'Total_Anomalous'}, inplace=True)

    # Convert columns to integers
    anomalousData['Total_Anomalous'] = anomalousData['Total_Anomalous'].astype(int)

    fig, ax = plt.subplots(figsize=(20, 12))  # Increased figure size

    # Plot the bar chart for anomalous records
    bars = ax.barh(np.arange(len(anomalousData)), anomalousData['Total_Anomalous'], color='#4B0082', label='Anomalous Records')

    # Set y-axis ticks and labels
    ax.set_yticks(np.arange(len(anomalousData)))
    ax.set_yticklabels(anomalousData[filter_Col], fontsize=25)

    # Annotate the bars with values inside the graph
    for bar in bars:
        width = bar.get_width()
        ax.text(width - 0.5, bar.get_y() + bar.get_height() / 2, str(int(width)), va='top', ha='left', color='black', fontsize=35)
    # Hide the x-axis
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.xaxis.set_visible(False)

    #ax.set_xlabel('Number of Records', fontsize=12)
    
    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    
    # Save or show the figure with tight bounding box to reduce extra white space
    fig.tight_layout(pad=0.5)
    plt.subplots_adjust(left=0.15, right=0.85) 
    return fig

def summary_changed():
    if st.session_state.summary_button==False:
        st.session_state.summary_button = True
        st.session_state.tabular_button = False

def train_changed():
    if st.session_state.button_train_model==False:
        st.session_state.button_train_model = True

def tabular_changed():
    if st.session_state.tabular_button==False:
        st.session_state.tabular_button = True
        st.session_state.summary_button = False                           
  
def main():
    selected_date_col=''
    st.markdown("""
        <style>
        .sidebar .sidebar-content {
            background-color: #FF0000;
        }
        .stApp {
            background-color: #E3F2FD;
        }
        .small-button {
            background-color: #FF0000;
            color: white;
            border: none;
            padding: 6px 12px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 14px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
        }
        .column-box {
            background-color: white;
            height: 80px;  /* Set the height of the div */
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 8px;
            text-align: center;
            color: black;
        }
        .button-container {
            display: flex;
            justify-content: flex-start;
            gap: 10px;
        }
        .date
        {
            font-style: italic;
            text-align: right;
            color: red;
        }
        </style>
        """, unsafe_allow_html=True)
    today = datetime.now()
    selected_period_col='R6M'
    if 'summary_button' not in st.session_state:
        st.session_state.summary_button = False
    if 'button_train_model' not in st.session_state:
        st.session_state.button_train_model = False
    
    if 'tabular_button' not in st.session_state:
        st.session_state.tabular_button = False

    # Format the date as dd-mm-yyyy
    formatted_date = today.strftime('%m/%d/%Y')
    st.markdown(f'<p class="date">Refresh Date:{formatted_date} </p>', unsafe_allow_html=True)
    
    # Retrieve file uploads
    file_type, yesterday_file, today_file = uploadedFiles()

    if today_file:
        if file_type == "CSV":
            today_df = pd.read_csv(today_file)
        elif file_type == "Excel":
            today_df = pd.read_excel(today_file)

        datetime_columns = get_datetime_columns(today_df)
        
        num_cols=today_df.select_dtypes(include=['int64','float64']).columns.to_list()
        Target_col=st.sidebar.selectbox("Select target Column",options=num_cols)
        #st.sidebar.subheader("Choose Date Columns")
        selected_date_col = st.sidebar.selectbox("Select date column", datetime_columns)
        #st.sidebar.subheader("Choose Period")
        min_date = today_df[selected_date_col].min()
        reference_date=today_df[selected_date_col].max()
        diff_months = (reference_date.year - min_date.year) * 12 + reference_date.month - min_date.month
        if diff_months<=6:
            period_options=["R6M"]
        elif diff_months<=12:
            period_options=["R6M","R12M"]
        elif diff_months<=12:
            period_options=["R6M","R12M","R18M"]
        else:
            period_options=["R6M","R12M","R18M","R24M"]
        selected_period_col = st.sidebar.selectbox("Select Period column", period_options)
        
        button_train_model=st.sidebar.button("Train Model",help="Train Model on Yesterday's Last "+selected_period_col,on_click=train_changed)
        
        
        
    if yesterday_file and today_file and st.session_state.button_train_model :
        
        button_container = st.container()
        with button_container:
            col1, col2 = st.columns([1, 2])
            with col1:
                summary_button = st.button("  Summary Dashboard  ", key="summary", help="Show summary dashboard",on_click=summary_changed)
            with col2:
                tabular_button = st.button("  Tabular Dashboard    " , key="tabular", help="Show tabular data",on_click=tabular_changed)
        
        if file_type == "CSV":
            data1 = pd.read_csv(yesterday_file)
            data2 = pd.read_csv(today_file)
        elif file_type == "Excel":
            data1 = pd.read_excel(yesterday_file)
            data2 = pd.read_excel(today_file)
                       
        print("Summary Button Status",st.session_state.summary_button)
        print("Train Button Status",st.session_state.button_train_model)            
        if st.session_state.summary_button:
        
            st.markdown(
            """
            <style>
            .title-font {
                color: black;
                font-size: larger;
            }
            </style>
            """,
            unsafe_allow_html=True
            )
            st.markdown('<p class="title-font"><b>Anomaly Detection in Input Data<b></p>',unsafe_allow_html=True)
            st.markdown('<p class="title-font">Along with comparison with data received in previous file<p>',unsafe_allow_html=True)

             
            
            
            period_number=int(re.findall(r'\d+', selected_period_col)[0])
            data1=filter_data_by_period(data1, selected_date_col, selected_period_col)
            #print
            testingdataML=timePlusBoosting(data1,data2,Target_col)
            testingdata=today_df.copy()
            testingdata.loc[:,'pred']=testingdataML['pred'].values
            print("Model Predicted Data\n",testingdata.head())

            results = compare_data(data1, data2, selected_date_col)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_rows_file1 = results['total_rows_file1']  # dynamically get the value
                st.markdown(
                    f"""
                     <style>
                            .fixed-size-div {{
                                width: 500px;  /* Set the width of the div */
                                height: 50px;  /* Set the height of the div */
                                padding: 10px;
                                border: 1px solid #ddd;
                                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
                                background-color: #f9f9f9;
                            }}
                     </style>
                        <div class='column-box' style='text-align: left;'>
                            <p style='text-align: center; color:black'><b> Rows in Previous File <b></p>
                            <p style='text-align: center; font-size: 150%;margin: -1rem'><b>{total_rows_file1}</b></p>
                            
                        </div>
                        """,
                    unsafe_allow_html=True
                )
                # st.markdown("<div class='column-box'><h4>Total Rows in Previous Data</h3></div>", unsafe_allow_html=True)
                # st.write(f"Total rows in previous file: {results['total_rows_file1']}")

            with col2:
                total_rows_file2 = results['total_rows_file2']  # dynamically get the value
                st.markdown(
                    f"""
                                        <div class='column-box' style='text-align: left;'>
                                            <p style='text-align: center; color:black'><b>Rows in Recent Data<b></p> 
                                            <p style='text-align: center; font-size: 150%;margin: -1rem'><b>{total_rows_file2}</b></p>
                                        </div>
                                        """,
                    unsafe_allow_html=True
                )
            with col3:
                total_rows_file1 = testingdata[testingdata['pred']==True].shape[0]  # dynamically get the value
                print("Number of Anomolous Data",total_rows_file1)
                st.markdown(
                    f"""
                                        <div class='column-box' style='text-align: left;'>
                                            <p style='text-align: center; color:black'><b>Anomalies Detected <b></p>
                                            <p style='text-align: center; font-size: 150%;margin: -1rem'><b>{total_rows_file1}</b></p>

                                        </div>
                                        """,
                    unsafe_allow_html=True
                )

            with col4:
                total_rows_file1 = results['Changes_in_old_File'].shape[0]  # dynamically get the value
                st.markdown(
                    f"""
                                        <div class='column-box' style='text-align: left;'>
                                            <p style='text-align: center; color:black'><b>Restatements<b></p>
                                            <p style='text-align: center; font-size: 150%;margin: -1rem'><b>{total_rows_file1}</b></p>

                                        </div>
                                        """,
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)

            col5, col6 = st.columns([2, 1])  # Adjust the width ratio

            with col5:
                data1['Response'] = 0 
                fig = brand_Comparison_Over_Time(data1, data2, Target_col)
                img_base64 = fig_to_base64(fig)
                #fig.set_size_inches(10, 4)

                st.markdown(
                    """
                    <style>
                    .Graph-box {
                        border: 2px solid #ddd;
                        border-radius: 5px;
                        padding: 20px;
                        margin: 20px 0;
                        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
                        background-color: #f9f9f9;
                        text-align: center;
                        padding: 20px;
                        margin: 25px 0;
                        height: 350px;
                    }
                    .Graph-box p {
                        font-size: 15px;
                        font-weight: bold;
                        color: #333;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(f"""
                    <div class='Graph-box'>
                        <p>Comparison Graph of Brand Level Distribution</p>
                        <img src='data:image/png;base64,{img_base64}' style='width:100%; height: 80%;'>
                    </div>
                    """, unsafe_allow_html=True)
            with col6:
                fig = BarChartNumberAnomaliesDetected(testingdata)
                img2_base64 = fig_to_base64(fig)

                st.markdown(
                    """
                    <style>
                    .Graph-box6 {
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
                        background-color: #f9f9f9;
                        text-align: center;
                        padding: 20px;
                        margin: 25px 0;
                        height: 350px;  /* Set a fixed height */
                    }
                    .Graph-box6 p {
                        font-size: 12px;
                        font-weight: bold;
                        color: #333;
                    }
                    .legend-box {
                        display: flex;
                        justify-content: center;
                        margin-top: 10px;
                    }
                    .legend-box div {
                        display: flex;
                        align-items: center;
                        margin-right: 20px;
                    }
                    .legend-box span {
                        width: 12px;
                        height: 12px;
                        display: inline-block;
                        margin-right: 5px;
                    }
                    .legend-box .green {
                        background-color: #D8BFD8;
                    }
                    .legend-box .red {
                        background-color: #4B0082;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"""
                    <div class='Graph-box6'>
                        <p>Total No. of Anomalies Detected in Current Data Load</p>
                        <img src='data:image/png;base64,{img2_base64}' style='width:100%'> 
                        <div class='legend-box'>
                            <div style="color:black" ><span class='green'></span><b>Non-Anomalous Records<b></div>
                            <div style="color:black" ><span class='red'></span><b>Anomalous Records<b></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                                        
            col7,col7_1=st.columns(2)
            with col7_1:
                filter_col='Brnd_Name'
                cat_cols=testingdata.select_dtypes(include=['object']).columns.to_list()
                print("Fiter Categorical columns",cat_cols)
                filter_col=st.selectbox("Brand",options=cat_cols)
                fig=Anomalies_Brand(testingdata,filter_col)  
                    #st.pyplot(fig)
                plt.show()
                img3_base64 = fig_to_base64(fig)
                st.markdown(
                    """
                    <style>
                    .Graph-box3 {
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);
                        background-color: #f9f9f9;
                        text-align: center;
                        padding: 20px;
                        margin-bottom:2rem ;  /* Remove margin for alignment */
                        gap:0;
                    }
                    .Graph-box3 p {
                        font-size: 15px;
                        font-weight: bold;
                        color: #333;
                    }
                    .image {
                        display: block;
                        width: 100%;
                        height: 275px;  /* Fixed height */
                        object-fit: cover; /* Ensures the image fills the space without distortion */
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )


                st.markdown(f"""<div class='Graph-box3'>
                    <p>Anomalies by {filter_col}</p>
                    <img class='image' src='data:image/png;base64,{img3_base64}'>
                    </div>""",
                    unsafe_allow_html=True
                ) 
                        
                       
            col8,col9=st.columns(2)
            with col8:
                st.markdown('<p style="color:black;font-size: 15px;font-weight: bold;" >Comparison with previous file (Tabular)</p>', unsafe_allow_html=True)
                display_comparison_results(results, data1, data2,Target_col, "Previous File", "Current File")
                # st.write('new_entries')
                # new_entries = find_new_entries(data1, data2)
                # display_new_entries(new_entries, "Previous File", "Current File")
            with col9:
                st.markdown(f'''<p style="color:black;font-size: 15px;font-weight: bold;" >Anamolies ranges at Brand level </p>''', unsafe_allow_html=True)
                
                if  filter_col=='Prscrbr_Type':
                    des_df=Data_Profile(testingdata, filter_col,Target_col)
                    
                else:
                     des_df=Data_Profile(testingdata, filter_col,Target_col)  
                
                       
                # des_df=Data_Profile(testingdata, 'Brnd_Name')        
                #time.sleep(5)     
                st.dataframe(des_df)        
        if st.session_state.button_train_model and st.session_state.tabular_button:
            if file_type == "CSV":
                yesterday_df = pd.read_csv(yesterday_file)
                today_df = pd.read_csv(today_file)
            elif file_type == "Excel":
                yesterday_df = pd.read_excel(yesterday_file)
                today_df = pd.read_excel(today_file)

            st.markdown('<p class="title-font" style="color:red" >Restatements of Previous Data Load And Current Data Load</p>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<p class="title-font" style="color:black" ><b>Previous Load<b></p>', unsafe_allow_html=True)
                results = compare_data(yesterday_df, today_df, selected_date_col)
                st.dataframe(results['Changes_in_old_File'])

            with col2:
                st.markdown('<p class="title-font" style="color:black" ><b>Recent Load<b></p>', unsafe_allow_html=True)
                st.dataframe(results['Changes_in_recent_File'])
            st.session_state.tabular_button=False
            
            st.markdown('<p class="title-font" style="color:black" ><b>New Rows in Recent file<b></p>', unsafe_allow_html=True)
            if results['new_records'].shape[0] ==0:
                st.markdown('<p  style="color:black" ><i>There is no new rows inserted in  the Recent file</i></p>', unsafe_allow_html=True)
            else:
                st.dataframe(results['new_records'])

            
            st.markdown('<p class="title-font" style="color:black" ><b>Deleted Rows from Recent file<b></p>', unsafe_allow_html=True)
            if results['deleted_records'].shape[0] ==0:
                st.markdown('<p  style="color:black" ><i>There is no rows deleted from the Recent file</i></p>', unsafe_allow_html=True)
            else:
                st.dataframe(results['deleted_records'])

            
    else:
        st.markdown(
            """
            <style>
            .title-font {
                color: black;
            }
            
            </style>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown('<p class="title-font" >Please upload both files to proceed</p>', unsafe_allow_html=True)
        st.markdown('<h1 class="title-font">Anomaly Detection Dashboard</h1>', unsafe_allow_html=True)  
        


if __name__ == "__main__":
    main()
