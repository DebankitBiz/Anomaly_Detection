import streamlit as st
import pandas as pd
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import io
import base64 
import time
from dateutil.relativedelta import relativedelta
from prophet import Prophet

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
    today = datetime.now().date()

    # Ensure the date column is in datetime format
    data1[date_col] = pd.to_datetime(data1[date_col]).dt.date
    data2[date_col] = pd.to_datetime(data2[date_col]).dt.date

    # Exclude today's data from the comparison
    data2_excluding_today = data2[data2[date_col] != today]

    # Total rows in each file
    total_rows_file1 = len(data1)
    total_rows_file2 = len(data2_excluding_today)

    # Identify rows that are completely missing in file 2 compared to file 1
    missing_in_file2 = data1[~data1.apply(tuple, 1).isin(data2_excluding_today.apply(tuple, 1))]

    # Identify rows that have changed between file 1 and file 2
    merged = pd.merge(data1, data2_excluding_today, on=list(data1.columns), how='outer', indicator=True)
    changes = merged[merged['_merge'] == 'right_only']
    changes = changes.iloc[:, :len(data1.columns)]  # Remove the extra columns

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
        'changes': changes,
        'missing_in_file2': missing_in_file2,
        'column_comparison': column_comparison
    }


# Function to display comparison results

@st.cache_data
def display_comparison_results(results, data1, data2, file1_name, file2_name):
    data1 = data1.copy()
    data2 = data2.copy()

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
    Comparison_df = pd.DataFrame({
    "Analysis": [
        "Addition of New Records",
        "Changes detected between the two files",
        "Deletion of records",
        "New Brands"
    ],
    "Value": [
        new_rows_count,
        results['changes'].shape[0],
        results['missing_in_file2'].shape[0],
        Delta_Brand
    ]
    })

    st.dataframe(Comparison_df)    
        



# Function to find new entries
def find_new_entries(data1, data2, removeColumns=['Value']):
    new_entries = {}
    yesterdayColumns = set(data1.columns)
    todayColumns = set(data2.columns)

    for removeCol in removeColumns:
        yesterdayColumns.remove(removeCol)

    for column in yesterdayColumns:
        if column in data1.columns and column in data2.columns:
            unique_df1 = set(data1[column].unique())
            unique_df2 = set(data2[column].unique())
            new_entries[column] = unique_df2 - unique_df1
        else:
            new_entries[column] = 'Column not found in both dataframes'

    return new_entries



# Function to display new entries
def display_new_entries(new_entries, file1_name, file2_name):
    for column, new_vals in new_entries.items():
        if isinstance(new_vals, set) and len(new_vals) > 0:
            st.write(f"\nNew {column}s in {file2_name} not present in {file1_name}:")
            for val in new_vals:
                st.write(f"- {val}")
        elif isinstance(new_vals, str):
            st.write(f"\n{column}: {new_vals}")
        else:
            st.write(f"\nNo new {column}s found.")

@st.cache_resource           
def timePlusBoosting(trainingdata,testingdata):
    data = trainingdata.copy()
    data = data[(data['Date'] >= '2023-04-01')].reset_index()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.rename(columns={'Date': 'ds', 'Tot_30day_Fills': 'y'})
    
    # Create empty DataFrames to store thresholds
    normal_thresholds_df = pd.DataFrame(columns=['Brnd_Name', 'upper_threshold', 'lower_threshold'])
    thresholds_df_std = pd.DataFrame(columns=['Brnd_Name', 'upper_threshold', 'lower_threshold'])
    
    brandList = data['Brnd_Name'].unique()
    
    for brandName in brandList:
        brand_data = data[data['Brnd_Name'] == brandName].copy()
        brand_data = brand_data.sort_values(by='ds')
        
        model = Prophet()
        model.fit(brand_data)
        
        future = model.make_future_dataframe(periods=1, freq='D')
        forecast = model.predict(future)
        
        merged = brand_data.merge(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left')
        
        merged['error'] = merged['y'] - merged['yhat']
        merged['uncertainty'] = merged['yhat_upper'] - merged['yhat_lower']
        
        ############################
        merged_std = merged.copy()
        
        window_size = 30  # You can adjust this window size
        merged_std['rolling_mean'] = merged_std['y'].rolling(window=window_size).mean()
        merged_std['rolling_std'] = merged_std['y'].rolling(window=window_size).std()
        
        # Dynamic thresholds
        merged_std['dynamic_upper_threshold'] = merged_std['rolling_mean'] + 2 * merged_std['rolling_std']
        merged_std['dynamic_lower_threshold'] = merged_std['rolling_mean'] - 2 * merged_std['rolling_std']
        
        merged_new_std = merged_std[['ds', 'Brnd_Name', 'y', 'yhat', 'dynamic_lower_threshold', 'dynamic_upper_threshold']]
        
        upper_threshold_std = merged_new_std['dynamic_upper_threshold'].mean()
        lower_threshold_std = merged_new_std['dynamic_lower_threshold'].mean()
        
        brand_thresholds_std = pd.DataFrame([{
            'Brnd_Name': brandName,
            'upper_threshold': upper_threshold_std,
            'lower_threshold': lower_threshold_std
        }])
        
        thresholds_df_std = pd.concat([thresholds_df_std, brand_thresholds_std], ignore_index=True)
        
        #############################
        
        merged_new = merged[['ds', 'Brnd_Name', 'y', 'yhat', 'yhat_lower', 'yhat_upper']]
        
        upper_threshold = merged_new['yhat_upper'].mean()
        lower_threshold = merged_new['yhat_lower'].mean()
        
        brand_thresholds_normal = pd.DataFrame([{
            'Brnd_Name': brandName,
            'upper_threshold': upper_threshold,
            'lower_threshold': lower_threshold
        }])
        
        normal_thresholds_df = pd.concat([normal_thresholds_df, brand_thresholds_normal], ignore_index=True)

    
    yesterday_df = trainingdata.copy()
    today_df = testingdata.copy()

    yesterday_df = yesterday_df[['Date','Brnd_Name','Tot_30day_Fills']]
    today_df = today_df[['Date','Brnd_Name','Tot_30day_Fills']]
    
    yesterday_df = yesterday_df[yesterday_df['Date'] >= '2023-04-01'].reset_index()
    yesterday_df.drop(['index'],axis=1,inplace=True)
    
    today_df = today_df[today_df['Date'] >= '2023-05-01'].reset_index()
    today_df.drop(['index'],axis=1,inplace=True)

    # One-hot encoding
    yesterday_df_encoded = pd.get_dummies(yesterday_df, columns=['Brnd_Name'])
    today_df_encoded = pd.get_dummies(today_df, columns=['Brnd_Name'])
    
    today_df_encoded = today_df_encoded.reindex(columns=yesterday_df_encoded.columns, fill_value=0)
    
    
    X_train = yesterday_df_encoded.drop(['Date', 'Tot_30day_Fills'], axis=1)
    y_train = yesterday_df_encoded['Tot_30day_Fills']
    
    
    X_test = today_df_encoded.drop(['Date', 'Tot_30day_Fills'], axis=1)
    y_test = today_df_encoded['Tot_30day_Fills']
    
    
    gbr = GradientBoostingRegressor(random_state=42)
    gbr.fit(X_train, y_train)
    
    
    y_pred = gbr.predict(X_test)
    today_df['predicted']=y_pred
    residuals = np.abs(y_test - y_pred)
    today_df['residuals']=residuals
    
    merged_df = pd.merge(today_df, thresholds_df_std, on=['Brnd_Name'], how='left')
    merged_df['pred']=0
    merged_df.loc[(merged_df['residuals'] > merged_df['upper_threshold']) | (merged_df['residuals'] < merged_df['lower_threshold']), 'pred'] = 1  

    
    return merged_df, thresholds_df_std, normal_thresholds_df
 
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
    testingdata['Date'] = pd.to_datetime(testingdata['Date'])

    if testingdata['Date'].dt.year.nunique() > 1:
        testingdata['time_period'] = testingdata['Date'].dt.year
    elif testingdata['Date'].dt.to_period('M').nunique() > 1:
        testingdata['time_period'] = testingdata['Date'].dt.to_period('M')
    else:
        testingdata['time_period'] = testingdata['Date'].dt.to_period('D')

    testing_grouped = testingdata.groupby(['time_period', 'Brnd_Name'])[Target_Col].sum().reset_index()

    # Merge the training and testing data on time_period and brand
    merged_data = pd.merge(training_grouped, testing_grouped, on=['time_period', 'Brnd_Name'], suffixes=('_train', '_test'))

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
        ax.text(width - 0.5, bar.get_y() + bar.get_height() / 2, str(int(width)), va='center', ha='right', color='black', fontsize=40)
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
        
        st.sidebar.subheader("Choose Date Columns")
        selected_date_col = st.sidebar.selectbox("Select date column", datetime_columns)
        st.sidebar.subheader("Choose Period")
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
            testingdata, thresholds_df_std, normal_thresholds_df=timePlusBoosting(data1,data2)
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
                total_rows_file1 = testingdata[testingdata['pred']==1].shape[0]  # dynamically get the value
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
                total_rows_file1 = results['total_rows_file1']  # dynamically get the value
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
                fig = brand_Comparison_Over_Time(data1, data2, 'Tot_30day_Fills')
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
            with col7:
                filter_col='Brnd_Name'
                filter_col=st.selectbox("Brand",options=['Brnd_Name','Indication'])
                
                if filter_col=='Índication':
                    fig=Anomalies_Brand(testingdata,filter_col) 
                else:
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
                        
                       
            col8=st.columns(1)[0]
            with col8:
                st.markdown('<p style="color:black;font-size: 15px;font-weight: bold;" >Comparison with previous file (Tabular)</p>', unsafe_allow_html=True)
                display_comparison_results(results, data1, data2, "Previous File", "Current File")
                # st.write('new_entries')
                # new_entries = find_new_entries(data1, data2)
                # display_new_entries(new_entries, "Previous File", "Current File")
            col9=st.columns(1)[0]    
            with col9:
                st.markdown(f'''<p style="color:black;font-size: 15px;font-weight: bold;" >Anamolies ranges at Brand level </p>''', unsafe_allow_html=True)
                
                if  filter_col=='Prscrbr_Type':
                    des_df=Data_Profile(testingdata, filter_col,'Tot_30day_Fills')
                    
                else:
                     des_df=Data_Profile(testingdata, filter_col,'Tot_30day_Fills')  
                des_df=pd.merge(des_df, thresholds_df_std, on=['Brnd_Name'])
                des_df=pd.merge(des_df, normal_thresholds_df, on=['Brnd_Name'])
                       
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
                st.markdown('<p class="title-font" style="color:black" ><b>Deleted from the Current File<b></p>', unsafe_allow_html=True)
                results = compare_data(yesterday_df, today_df, selected_date_col)
                st.dataframe(results['missing_in_file2'])

            with col2:
                st.markdown('<p class="title-font" style="color:black" ><b>Changes between two files<b></p>', unsafe_allow_html=True)
                st.dataframe(results['changes'])
            st.session_state.tabular_button=False

            
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
