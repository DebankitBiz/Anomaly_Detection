import streamlit as st
import pandas as pd
from datetime import datetime,timedelta,date
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import io
import base64 
import time
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
import calendar

from sklearn.base import BaseEstimator, TransformerMixin



class HandleUnknownCategories(BaseEstimator, TransformerMixin):
    def __init__(self, known_categories=None):
        self.known_categories = known_categories

    def fit(self, X, y=None):
        # Store unique categories from training data
        self.known_categories = {col: set(X[col].unique()) for col in X.columns}
        return self

    def transform(self, X):
        # Map unseen categories to "Unknown"
        X_transformed = X.copy()
        for col in X.columns:
            X_transformed[col] = X[col].apply(lambda x: x if x in self.known_categories[col] else "Unknown")
        return X_transformed



class EmptyDataFrameError(Exception):
    """Exception raised when the DataFrame is empty."""
    def __init__(self, message="The DataFrame is empty."):
        self.message = message
        super().__init__(self.message)
        
def highlight_pass_fail(val):
    if val == 'Pass':
        color = 'White'
        text_color = 'Green'
        symbol = '\u25cf'  # Green filled circle
    elif val == 'Fail':
        color = 'White'
        text_color = 'Red'
        symbol = '\u25cf'  # Red filled circle
    elif val == 'Pass.':
        color = 'White'
        text_color = '#ED7D31'
        symbol = '\u25cf'  # Red filled circle
    else:
        color = 'white'
        text_color = 'black'
        symbol = '\u25cf'  # Black filled circle
    return f'background-color: {color}; color: {text_color}; text-align: center; padding: 2px 0px 2px 0px; content: "{symbol}"'
  



def set_page_config():
    PAGE_CONFIG = {
        "page_title": "Data Observality",
        "page_icon":"",
        "layout": "wide",
        "initial_sidebar_state": "expanded",
    }
    st.set_page_config(**PAGE_CONFIG)

set_page_config()


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
def MonthSummary(df,brand_selection):
    df=df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df = df[df['Brnd_Name'].isin(brand_selection)]
    numerical_col=['Tot_Clms', 'Tot_Drug_Cst','Tot_Day_Suply','Tot_30day_Fills']
    monthly_summary = df.groupby('Month')[numerical_col].sum().reset_index()
    monthly_summary[numerical_col] = monthly_summary[numerical_col].astype(int)
    monthly_summary.index=monthly_summary.index+1
    st.dataframe(monthly_summary)




def uploadedFiles():
    
    st.sidebar.markdown("""
            <style>
            .sidebar .sidebar-content {
                background-color: #373D50; /* Yellow background */
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
        yesterday_file = st.sidebar.file_uploader("Upload Last file (Monthly/Weekly)", type=["csv", "xlsx"])
    today_file = st.sidebar.file_uploader("Upload Current file (Monthly/Weekly)", type=["csv", "xlsx"])
    return file_type, yesterday_file, today_file

def get_datetime_columns(df):
    # Get columns with datetime format
    datetime_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    return datetime_columns


@st.cache_data
def structural_change(yesterday_df, today_df):
    
    # Compare numerical columns
    yesterday_numerical = set(yesterday_df.select_dtypes(include=['int', 'float']).columns)
    today_numerical = set(today_df.select_dtypes(include=['int', 'float']).columns)
    added_columns = today_numerical - yesterday_numerical
    removed_columns = yesterday_numerical - today_numerical
    status_num = "Fail" if added_columns or removed_columns else "Pass"
    comment_num = f"Added: {', '.join(added_columns)}. Removed: {', '.join(removed_columns)}." if added_columns or removed_columns else "No numerical columns added or removed."
    
    # Create DataFrame for numerical columns
    df_num = pd.DataFrame({"Status": [status_num], "Comment": [comment_num]}, index=['Numerical Columns'])  
    
    # Compare categorical columns
    yesterday_categorical = set(yesterday_df.select_dtypes(include=['object']).columns)
    today_categorical = set(today_df.select_dtypes(include=['object']).columns)
    added_columns = today_categorical - yesterday_categorical
    removed_columns = yesterday_categorical - today_categorical
    status_cat = "Fail" if added_columns or removed_columns else "Pass"
    comment_cat = f"Added: {', '.join(added_columns)}. Removed: {', '.join(removed_columns)}." if added_columns or removed_columns else "No categorical columns added or removed."
    
    # Create DataFrame for categorical columns
    df_cat = pd.DataFrame({"Status": [status_cat], "Comment": [comment_cat]}, index=['Categorical Columns'])
    
    # Compare date columns
    yesterday_date = set(yesterday_df.select_dtypes(include=['datetime', 'datetime64']).columns)
    today_date = set(today_df.select_dtypes(include=['datetime', 'datetime64']).columns)
    added_columns = today_date - yesterday_date
    removed_columns = yesterday_date - today_date
    status_date = "Fail" if added_columns or removed_columns else "Pass"
    comment_date = f"Added: {', '.join(added_columns)}. Removed: {', '.join(removed_columns)}." if added_columns or removed_columns else "No date columns added or removed."
    
    # Create DataFrame for date columns
    df_date = pd.DataFrame({"Status": [status_date], "Comment": [comment_date]}, index=['Date Columns'])
    
    # Concatenate all DataFrames
    df = pd.concat([df_num, df_cat, df_date])
  
    styled_df = df.style.applymap(highlight_pass_fail)
    
    st.dataframe(styled_df, width=1500) 




def restatement_changed():
    if st.session_state.restatement_button==False:
        st.session_state.restatement_button=True
        st.session_state.anomaly_button = False   


def anomaly_changed():
    if st.session_state.anomaly_button==False:
        st.session_state.restatament_button=False
        st.session_state.anomaly_button = True
        
def train_changed():
    if st.session_state.button_train_model==False:
        st.session_state.button_train_model = True        
             

def add_commas(number):
    # Convert the number to a string and format it with commas
    return "{:,}".format(number)


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
    #print("Inside Compare Data",total_rows_file2)

    categorical_columns = data2_filtered.select_dtypes(include=['object']).columns.to_list()
    categorical_columns.append(date_col)
    categorical_columns=['Date','Prscrbr_NPI','Brnd_Name']
    
    old_tuples = data1[categorical_columns].apply(tuple, axis=1)
    new_tuples = data2_filtered[categorical_columns].apply(tuple, axis=1)
    
    new_records = data2_filtered[~new_tuples.isin(old_tuples)].reset_index(drop=True)
    new_records.index = new_records.index + 1
    
    
    deleted_records = data1[~old_tuples.isin(new_tuples)].reset_index(drop=True)
    deleted_records.index = deleted_records.index + 1
    
    # Identify rows that are completely missing in file 2 compared to file 1
    Changes_in_recent_File=data2_filtered[~data2_filtered.apply(tuple, 1).isin(data1.apply(tuple, 1))]
    Changes_in_old_File = data1[~data1.apply(tuple, 1).isin(data2_filtered.apply(tuple, 1))]
    
    
    Changes_in_recent_File=Changes_in_recent_File[~Changes_in_recent_File.apply(tuple,1).isin(new_records.apply(tuple,1))]
    Changes_in_old_File=Changes_in_old_File[~Changes_in_old_File.apply(tuple,1).isin(deleted_records.apply(tuple,1))]
    
    Changes_in_old_File=Changes_in_old_File.reset_index(drop=True)
    Changes_in_old_File.index=Changes_in_old_File.index+1

    Changes_in_recent_File=Changes_in_recent_File.reset_index(drop=True)
    Changes_in_recent_File.index=Changes_in_recent_File.index+1
    
    Changes_in_recent_File_new=Changes_in_recent_File.copy()
   
    Changes_in_recent_File_new['Changed'] = ''

    # Find elements with changes and update the 'Changed' column
    for i in range(1,len(Changes_in_old_File)+1):
        changes=[]
        for col in Changes_in_old_File.columns:
             
            if Changes_in_old_File.loc[i, col] != Changes_in_recent_File.loc[i, col] and not (pd.isna(Changes_in_old_File.loc[i, col]) and pd.isna(Changes_in_recent_File.loc[i, col])):
                changes.append(f"{col} has changes from  {Changes_in_old_File.loc[i, col]} to {Changes_in_recent_File.loc[i, col]}")
        changes_str = '; '.join(changes)
        Changes_in_recent_File_new.loc[i,'Changed']=changes_str

        

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
        'deleted_records':deleted_records,
        'Changes_in_recent_File_new':Changes_in_recent_File_new
    }




def categorize_residual(residual, anomaly,max_residual):
    

    high_threshold = 0.5 * max_residual  # 50% of max residual
    medium_threshold = 0.3 * max_residual  # 30% of max residual
    if not anomaly:
        return 'Not Anomalous'
    elif residual > high_threshold:
        return 'Significant'
    elif residual > medium_threshold:
        return 'Moderate'
    else:
        return 'Minor'

@st.cache_resource           
def timePlusBoosting(trainingdata,testingdata,target_col,from_month):
    try :
        
        trainingdata=trainingdata.copy()
        testingdata=testingdata.copy()
        testingdata.loc[:,'pred']=0
        
        testingdata=testingdata[testingdata['Date'].dt.month >= from_month].reset_index(drop=True)
        #testingdata = testingdata[testingdata['Date'] >= '2023-04-01'].reset_index(drop=True)
        
        yesterday_df_filtered = trainingdata[[ 'Date','Prscrbr_NPI' ,'Prscrbr_State_Abrvtn', 'Brnd_Name', 'Tot_Clms', 'Tot_Day_Suply', 'Tot_Drug_Cst', 'Tot_30day_Fills']]
        today_df_filtered = testingdata[[ 'Date', 'Prscrbr_NPI','Prscrbr_State_Abrvtn', 'Brnd_Name', 'Tot_Clms', 'Tot_Day_Suply', 'Tot_Drug_Cst', 'Tot_30day_Fills']]
        
        categorical_cols = ['Brnd_Name', 'Prscrbr_State_Abrvtn']
        transformer = HandleUnknownCategories()
        yesterday_categoricals_transformed = transformer.fit_transform(yesterday_df_filtered[categorical_cols])
        today_categoricals_transformed = transformer.transform(today_df_filtered[categorical_cols])
        
        yesterday_df_filtered[categorical_cols] = yesterday_categoricals_transformed
        today_df_filtered[categorical_cols] = today_categoricals_transformed
        
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        # Fit on yesterday's data and transform both yesterday's and today's data
        yesterday_encoded = encoder.fit_transform(yesterday_df_filtered[categorical_cols])
        today_encoded = encoder.transform(today_df_filtered[categorical_cols])
        
        # Get feature names and convert encoded arrays to DataFrames
        encoded_columns = encoder.get_feature_names_out(categorical_cols)
        yesterday_df_encoded = pd.DataFrame(yesterday_encoded, columns=encoded_columns, index=yesterday_df_filtered.index)
        today_df_encoded = pd.DataFrame(today_encoded, columns=encoded_columns, index=today_df_filtered.index)

        numerical_cols = ['Tot_Clms', 'Tot_Day_Suply', 'Tot_Drug_Cst']

        # Add Date and Prscrbr_NPI back to the DataFrames
        yesterday_df_encoded = pd.concat([yesterday_df_filtered[['Date', 'Prscrbr_NPI','Tot_30day_Fills']], yesterday_df_encoded, yesterday_df_filtered[numerical_cols]], axis=1)
        today_df_encoded = pd.concat([today_df_filtered[['Date', 'Prscrbr_NPI','Tot_30day_Fills']], today_df_encoded, today_df_filtered[numerical_cols]], axis=1)
        
        scaler = MinMaxScaler()
        yesterday_df_encoded[numerical_cols] = scaler.fit_transform(yesterday_df_encoded[numerical_cols])
        today_df_encoded[numerical_cols] = scaler.transform(today_df_encoded[numerical_cols])
        
        # Splitting the data into features and target
        X_train = yesterday_df_encoded.drop([ 'Date','Prscrbr_NPI', 'Tot_30day_Fills'], axis=1)
        y_train = yesterday_df_encoded[target_col]
        X_test = today_df_encoded.drop([ 'Date', 'Prscrbr_NPI','Tot_30day_Fills'], axis=1)
        y_test = today_df_encoded[target_col]

        # Gradient Boosting Regressor model
        gbr = GradientBoostingRegressor(random_state=42)
        gbr.fit(X_train, y_train)

        # Predictions
        y_pred = gbr.predict(X_test)
        today_df_filtered['predicted'] = y_pred
        residuals = np.abs(y_test - y_pred)
        today_df_filtered['residuals'] = residuals
        today_df_filtered['anomaly'] = False
        
        today_df_filtered['anomaly'] = today_df_filtered.apply(
            lambda row: any(row[col] == "Unknown" for col in categorical_cols),
            axis=1
        )
        #today_df_filtered.to_csv("Unknown Conversion.csv")
        today_df_filtered['month'] = pd.to_datetime(today_df_filtered['Date']).dt.to_period('M')
        # Ensure all rows are initially marked as non-anomalous
  

        # Threshold Calculation
        threshold_data = []
        for brand in today_df_filtered['Brnd_Name'].unique():
            for state in today_df_filtered['Prscrbr_State_Abrvtn'].unique():
                for month in today_df_filtered['month'].unique():
                    # Filter the DataFrame for the current brand, state, and month
                    brand_state_month_df = today_df_filtered[
                        (today_df_filtered['Brnd_Name'] == brand) & 
                        (today_df_filtered['Prscrbr_State_Abrvtn'] == state) &
                        (today_df_filtered['month'] == month)
                    ].reset_index(drop=True)
                    
                    if not brand_state_month_df.empty:
                        # Calculate the median and MAD for the residuals
                        median_residuals = np.median(brand_state_month_df['residuals'])
                        mad_residuals = np.median(np.abs(brand_state_month_df['residuals'] - median_residuals))

                        # Calculate upper and lower thresholds
                        upper_threshold = median_residuals + 3 * mad_residuals
                        lower_threshold = median_residuals - 3 * mad_residuals

                        # Append the brand, state, month, and their thresholds to the list
                        threshold_data.append({
                            'Brnd_Name': brand, 
                            'Prscrbr_State_Abrvtn': state,
                            'month': month,
                            'upper_residual_threshold': upper_threshold, 
                            'lower_residual_threshold': lower_threshold
                        })

                        # Mark anomalies based on the thresholds for the current brand, state, and month
                        today_df_filtered.loc[
                            (today_df_filtered['Brnd_Name'] == brand) & 
                            (today_df_filtered['Prscrbr_State_Abrvtn'] == state) &
                            (today_df_filtered['month'] == month) &
                            ((today_df_filtered['residuals'] > upper_threshold) | 
                            (today_df_filtered['residuals'] < lower_threshold)), 
                            'anomaly'
                        ] = True
                        
        threshold_df = pd.DataFrame(threshold_data)

        # Merge threshold information back into today_df_filtered
        today_df_filtered = today_df_filtered.merge(threshold_df, how='left', on=['Brnd_Name', 'Prscrbr_State_Abrvtn', 'month'])

        # Reintegrate the original columns using "IDs" for merging
        today_df_final = testingdata.merge(today_df_filtered[[ 'predicted', 'residuals', 'anomaly']], 
                                how='left', 
                                left_index=True, 
                                right_index=True)
        #today_df_final.to_csv("After Merge.csv")

        # Extract anomalies
        anomaliesGetStd = today_df_final[today_df_final['anomaly'] == True].reset_index(drop=True)
        condition = (today_df_final['residuals'] < 5)

        today_df_final.loc[condition, 'anomaly'] = False

        max_residual = today_df_final['residuals'].max()
        today_df_final['residual_category'] = today_df_final.apply(
            lambda row: categorize_residual(row['residuals'], row['anomaly'],max_residual), axis=1
        )

        print("Check new Brand")
        unknown_brands = today_df_filtered['Brnd_Name'] == 'Unknown'

        # Set 'anomaly' to True and 'residuals' category to 'High'
        today_df_final.loc[unknown_brands, 'anomaly'] = True
        today_df_final.loc[unknown_brands, 'residual_category'] = 'Significant'
        #today_df_final.to_csv("Final testing data.csv")
        
        anomaly_indices = today_df_final[today_df_final['anomaly'] == True].index
        testingdata.loc[anomaly_indices,'pred']=1
        testingdata.loc[:,'residual_category']=today_df_final['residual_category']
        
        #testingdata.to_csv("testingdata.csv")
        
    except Exception as e:
        st.markdown(f'<p  style="color:black" ><i>{e}</i></p>', unsafe_allow_html=True)
            
    return testingdata,threshold_df

@st.cache_data
def display_comparison_results(results, data1, data2):
    try:
        
        data1 = data1.copy()
        data2 = data2.copy()
        data1, data2 = filter_data_by_date_range(data1, data2, 'Date') 
        
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
        new_records = results['new_records']
        
        Changes_in_recent_File = results['Changes_in_recent_File'].copy()
        Changes_in_old_File = results['Changes_in_old_File'].copy()
        if Changes_in_old_File.empty or Changes_in_recent_File.empty:
            raise EmptyDataFrameError()
            
        # Create a DataFrame to hold summary information
        summary_df = pd.DataFrame({
            "Title": [
                "Records Added in Recent File",
                "Records Deleted from the Recent file",
                "Records Modified in the Recent file",
                "New HCP added in the Recent file",
                "HCP deleted in the Recent file"
            ],
            "Value": [
                results['new_records'].shape[0],
                results['deleted_records'].shape[0],
                results['Changes_in_recent_File'].shape[0],
                new_records['Prscrbr_NPI'].nunique(),
                results['deleted_records']['Prscrbr_NPI'].nunique()
            ]
        })

        # # Define the key columns
        # key_columns = ['Date', 'Prscrbr_NPI', 'Brnd_Name']

        # # Create composite keys for both DataFrames
        # Changes_in_old_File['Composite_Key'] = Changes_in_old_File[key_columns].astype(str).agg(','.join, axis=1)
        # Changes_in_recent_File['Composite_Key'] = Changes_in_recent_File[key_columns].astype(str).agg(','.join, axis=1)

        # # Set the composite key as the index
        # Changes_in_old_File.set_index('Composite_Key', inplace=True)
        # Changes_in_recent_File.set_index('Composite_Key', inplace=True)

        # # Ensure both DataFrames have the same columns
        # common_columns = Changes_in_old_File.columns.intersection(Changes_in_recent_File.columns)
        # st.write(common_columns)

        # # Subset both DataFrames to include only the common columns
        # Changes_in_old_File = Changes_in_old_File[common_columns]
        # Changes_in_recent_File = Changes_in_recent_File[common_columns]
        # st.dataframe(Changes_in_old_File)
        # st.dataframe(Changes_in_recent_File)

        # # Compare the changes between the old and recent files
        # comparison = Changes_in_old_File != Changes_in_recent_File

        # # Display the comparison results
        # st.dataframe(comparison)

        # # Initialize a list to store change details
        # change_details = []

        # for index, row in comparison.iterrows():
        #     if row.any():  # If any column in the row is different
        #         mismatches = row[row].index.tolist()
        #         for col in mismatches:
        #             old_value = Changes_in_old_File.loc[index, col]
        #             new_value = Changes_in_recent_File.loc[index, col]
        #             date, NPI, Brand = index.split(",")
        #             if not (pd.isna(old_value) and pd.isna(new_value)):
        #                 change_details.append([
        #                     f"On {date} for NPI {NPI} and Brand {Brand}, Column {col}",
        #                     f"Value Changed from {old_value} to {new_value}"
        #                 ])

        # # Convert the change details to a DataFrame
        # changes_df = pd.DataFrame(change_details, columns=['Title', 'Value'])
        
        
        # # Display the changes DataFrame in Streamlit
        # if not changes_df.empty:
        #     summary_df=pd.concat([summary_df, changes_df],ignore_index=True) 

        

        # Identify and display new and deleted categorical values
        categorical_columns = data1.select_dtypes(include=['object']).columns.to_list()
        categorical_columns = ['Brnd_Name', 'Drug Type', 'Prscrbr_Type','Prscrbr_City','Prscrbr_State_Abrvtn']
        
        new_values_list = []
        del_values_list = []
        
        for col in categorical_columns:
            new_in_data2 = set(data2[col].astype(str).unique()) - set(data1[col].astype(str).unique())
            #st.write(new_in_data2)
            if new_in_data2:
                new_values_list.append(['In column '+col+' new values', ', '.join(new_in_data2)])
            
            del_in_data2 = set(data1[col].astype(str).unique()) - set(data2[col].astype(str).unique())
            #st.write(del_in_data2)
            if del_in_data2:
                del_values_list.append(['In column '+col+' deleted values', ', '.join(del_in_data2)])

        if new_values_list:
            new_values_df = pd.DataFrame(new_values_list, columns=['Title', 'Value'])
            summary_df=pd.concat([summary_df, new_values_df],ignore_index=True) 
        
        if del_values_list:
            del_values_df = pd.DataFrame(del_values_list, columns=['Title', 'Value'])
            summary_df=pd.concat([summary_df, del_values_df],ignore_index=True)
        
        # Display the summary DataFrame in Streamlit
        summary_df.index=summary_df.index+1
        st.dataframe(summary_df,width=1500)
    except EmptyDataFrameError as e:
        st.markdown('<p  style="color:black" ><i>There are no restatements in  the Recent file</i></p>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<p  style="color:black" ><i>{e}</i></p>', unsafe_allow_html=True)
        print(e)
        
        
    return 0




@st.cache_data
def Brand_Trend_Graph(df1, df2, selected_brand, month_select, Num_Dim):
    # Create copies of DataFrames to avoid modifying the originals
    df1 = df1.copy()
    df2 = df2.copy()
    
    # Create a mapping of month names to numbers
    month_name_to_num = {month: index for index, month in enumerate(calendar.month_name) if month}

    # Check if month_select contains numbers or names
    if month_select:
        # Ensure month_select contains month names, not numbers
        if isinstance(month_select[0], str):
            # Convert the selected month names to their corresponding numbers
            month_numbers = [month_name_to_num[month] for month in month_select]
        else:
            # Assume month_select already contains month numbers
            month_numbers = month_select
        
        # Filter both DataFrames based on the selected month numbers
        df1 = df1[df1['Date'].dt.month.isin(month_numbers)]
        df2 = df2[df2['Date'].dt.month.isin(month_numbers)]
        
    # Ensure selected_brand is a list
    if isinstance(selected_brand, str):
        selected_brand = [selected_brand]
    
    # Initialize a Plotly figure
    fig = go.Figure()
    
    # Loop through each brand to plot its data
    for brand in selected_brand:
        # Filter df1 for the current brand
        brand_data1 = df1[df1['Brnd_Name'] == brand]
        # Aggregate data by date for daily totals
        daily_aggregate1 = brand_data1.groupby('Date').agg({Num_Dim: 'sum'}).reset_index()

        # Add a trace for df1 (could be current data)
        fig.add_trace(go.Scatter(x=daily_aggregate1['Date'], y=daily_aggregate1[Num_Dim], 
                                 mode='lines', name=f'{brand} (Last File Month)'))

        # Filter df2 for the current brand
        brand_data2 = df2[df2['Brnd_Name'] == brand]
        daily_aggregate2 = brand_data2.groupby('Date').agg({Num_Dim: 'sum'}).reset_index()

        # Add a trace for df2 (could be historical data)
        fig.add_trace(go.Scatter(x=daily_aggregate2['Date'], y=daily_aggregate2[Num_Dim], 
                                 mode='lines', name=f'{brand} (Current File Month)', line=dict(dash='dash')))

    # Update layout to improve readability
    fig.update_layout(title={'text': "Brand Trend Over Time", 'x': 0.5, 'xanchor': 'center'}, xaxis_title="Date", yaxis_title=Num_Dim,
                      legend_title="Brand", legend=dict(yanchor="top", y=-0.3, xanchor="center", x=0.5))
    
    
    st.plotly_chart(fig, use_container_width=True)
    return fig


@st.cache_data
def Anomalies_Brand(testingData, filter_Col):
    # Filter and count anomalous records
    anomalousData = testingData[testingData['pred'] == 1].groupby([filter_Col])['pred'].count().reset_index()
    anomalousData.rename(columns={'pred': 'Total_Anomalous'}, inplace=True)

    # Sort by the total number of anomalies and get the top 10
    anomalousData = anomalousData.sort_values(by='Total_Anomalous', ascending=True).head(10)

    # Convert columns to integers
    anomalousData['Total_Anomalous'] = anomalousData['Total_Anomalous'].astype(int)

    # Determine text position based on Total_Anomalous value
    text_position = ['inside' if value >= 20 else 'outside' for value in anomalousData['Total_Anomalous']]  # Adjust threshold as needed

    # Create a Plotly bar chart
    fig = go.Figure()

    # Add bars to the figure
    fig.add_trace(go.Bar(
        y=anomalousData[filter_Col],
        x=anomalousData['Total_Anomalous'],
        orientation='h',
        marker=dict(color='#4B0082'),
        text=anomalousData['Total_Anomalous'],
        textposition=text_position,  # Set text position based on visibility
        textfont=dict(size=15)  # Increase data label size
    ))

    # Update layout for better appearance
    fig.update_layout(
        title='Top 5 Anomalous Records by {}'.format(filter_Col),
        xaxis_title='Number of Records',
        yaxis_title=filter_Col,
        yaxis=dict(tickfont=dict(size=15)),  # Reduce y-axis label size
        xaxis=dict(tickfont=dict(size=15)),
        showlegend=False,
        margin=dict(l=150, r=150, t=50, b=50),
        height=350  # Set height to 350 pixels
    )
    st.markdown(
        """
        <style>
        .graph-container {
            margin-top: 9px;  /* Adjust the margin as needed to shift the graph down */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        st.markdown('<div class="graph-container"></div>', unsafe_allow_html=True)  # Apply the custom styling
        st.plotly_chart(fig, use_container_width=True)


def fig_to_base64(_fig):
    buf = io.BytesIO()
    _fig.savefig(buf, format="png", bbox_inches='tight')  # Ensure entire figure is saved
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return img_base64


@st.cache_data
def BarChartNumberAnomaliesDetected(testingData):
    fig, ax = plt.subplots(figsize=(6, 1))  # Increased height to 3

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
    ax.text(goodRecords / 2, 0, str(goodRecords), va='center', ha='center', color='black', fontsize=15)
    ax.text(goodRecords + anomalousRecords / 2, 0, str(anomalousRecords), va='center', ha='center', color='black',
            fontsize=15)

    plt.tight_layout()

    # Move legend
    #plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), bbox_transform=ax.transAxes, fontsize=8)
    return fig

@st.cache_data
def plot_pie(df, select_col):
    # Ensure the 'Brnd_Name' column exists in the DataFrame
    
    # Filter DataFrame based on the selected brandsdf = df.copy()

    # Count values in the selected column
    select_col_counts = df[select_col].value_counts().reset_index()

    # Rename the columns
    select_col_counts.columns = [select_col, 'Count']

    # Sort the result by 'Count' and select the top 5
    select_col_counts = select_col_counts.sort_values(by='Count', ascending=False).head(5)
    
    if select_col=='Prscrbr_State_Abrvtn':
        title='State'
    elif select_col=='Prscrbr_Type':
        title='Speciality'
    elif select_col=='Brnd_Name':
        title='Brand'    
    elif select_col=='Prscrbr_City':
        title='City'
    else:
        title='Drug Type'            
        
    
    
    # Calculate percentages
    select_col_counts['Percentage'] = (select_col_counts['Count'] / select_col_counts['Count'].sum() * 100).round(1)
    select_col_counts['Legend_Label'] = select_col_counts[select_col] + ': ' + select_col_counts['Percentage'].astype(str) + '%'

    # Apply custom CSS to style the container
    st.markdown(
        """
        <style>
        .custom-container {
            background-color: white;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Plot the pie chart
    fig = px.pie(
        select_col_counts, 
        names='Legend_Label',  # Use the custom legend labels with percentages
        values='Count', 
        title=title+' Level Distribution',
        hover_data={'Count': True},  # Show count on hover
        labels={'Count': 'Data Points'}
    )

    # Customize hover data to show percentage
    fig.update_traces(
        textposition='inside', 
        textinfo='percent+label', 
        hovertemplate=title+': %{label}<br>Data Points: %{value}<br>Percentage: %{percent}'
    )

    # Customize the layout to center the title and adjust legend placement
    fig.update_layout(
        title={
            'text': title+' Level Distribution',
            'x': 0.5,  # Center the title horizontally
            'xanchor': 'center'  # Anchor the title at the center
        },
        legend=dict(
            orientation="h",  # Horizontal orientation
            yanchor="top",  # Anchor to the top
            y=-0.2,  # Position below the chart
            xanchor="center",  # Center the legend
            x=0.5  # Center alignment
        ),
        height=500,  # Set a fixed height for the plot to ensure consistency
        margin=dict(t=50, b=100)  # Adjust margins to fit the legend
    )

    # Display the plot in Streamlit with a smaller size inside a custom container
    st.plotly_chart(fig, use_container_width=True)



def main():
    selected_date_col=''
    st.markdown("""
        <style>
        .sidebar .sidebar-content {
            background-color: #373D50;
            color:white;
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
            color: gray;
            font-size:small;
        }
        </style>
        """, unsafe_allow_html=True)
    today = datetime.now()
    selected_period_col='R6M'


    if 'restatement_button' not in st.session_state:
        st.session_state.restatement_button = False
    if 'anomaly_button' not in st.session_state:
        st.session_state.anomaly_button=False
    if 'button_train_model' not in st.session_state:
        st.session_state.button_train_model = False
    
    
     # Format the date as dd-mm-yyyy
    formatted_date = today.strftime('%m/%d/%Y')
    st.markdown(f'<p class="date">Refresh Date: {formatted_date} </p>', unsafe_allow_html=True)
           
    
    
    file_type, yesterday_file, today_file = uploadedFiles()
    
    
    if today_file:
        if file_type == "CSV":
            today_df = pd.read_csv(today_file)
        elif file_type == "Excel":
            today_df = pd.read_excel(today_file)
    
        datetime_columns = get_datetime_columns(today_df)        
        num_cols=today_df.select_dtypes(include=['int64','float64']).columns.to_list()
        Target_col='Tot_30day_Fills'
        
        selected_date_col = st.sidebar.selectbox("Select date column", datetime_columns)
        
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
        
    if  yesterday_file and today_file:
        
        if file_type == "CSV":
            yesterday_df = pd.read_csv(yesterday_file)
            today_df = pd.read_csv(today_file)
        elif file_type == "Excel":
            yesterday_df = pd.read_excel(yesterday_file)
            today_df = pd.read_excel(today_file)
     
        month_select = st.sidebar.multiselect(
            "Select Month",
            options=today_df['Date'].dt.month.unique(),  # Get unique months
            format_func=lambda x: calendar.month_name[x],  # Convert month numbers to month names
            default=None  # You can set this to an empty list [] if you don't want any default selection
        )
        brand_options = list(set(yesterday_df['Brnd_Name'].unique()).union(set(today_df['Brnd_Name'].unique())))
        brand_selected=st.sidebar.multiselect("Brand",brand_options, default='Anoro Ellipta')
        
        period_number=int(re.findall(r'\d+', selected_period_col)[0])
        yesterday_df=filter_data_by_period(yesterday_df, selected_date_col, selected_period_col)
        
        #dimension_option=yesterday_df.select_dtypes(include=['object']).columns.to_list()
        Dimension_selection=st.sidebar.selectbox("Dimension",options=['Brnd_Name','Prscrbr_Type','Prscrbr_State_Abrvtn','Prscrbr_City','Drug Type'], index=0)
        
        Metric_selection=st.sidebar.selectbox("Select Metrics",options=['Tot_30day_Fills','Tot_Clms','Tot_Day_Suply','Tot_Drug_Cst'],index=0)
        
        
        anomalies_category=st.sidebar.multiselect("Anomaly Category",['Significant','Moderate','Minor'], default='Significant')
        
        st.sidebar.button("Train Model",help="Train Model on Yesterday's Last "+selected_period_col,on_click=train_changed)
        
        if yesterday_file and today_file and st.session_state.button_train_model :
            
            
            testingdata=today_df.copy()
            testingdata = testingdata[testingdata['Date'] >= '2023-04-01'].reset_index(drop=True)
            testingdataML,threshold_df=timePlusBoosting(yesterday_df,today_df,Target_col,4)
            testingdata[['residual_category', 'pred']] = testingdataML[['residual_category', 'pred']].values
            results = compare_data(yesterday_df, today_df, selected_date_col)
            
            
            print("Restatement",st.session_state.restatement_button)
            print("Anomaly",st.session_state.anomaly_button)
            
            
            button_container = st.container()
            with button_container:
                col1, col2 = st.columns(2)
                with col2:
                    st.button("  Anomaly   ", key="summary", help="Show summary dashboard",on_click=anomaly_changed)
                with col1:
                    st.button("  Overview    " , key="tabular", help="Show Restatements of data",on_click=restatement_changed)
            
            
            if st.session_state.anomaly_button:
        
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
                
                col1, col2, col3, col4 = st.columns(4)
            
                # st.markdown("<div class='column-box'><h4>Total Rows in Last Month Data</h3></div>", unsafe_allow_html=True)
                # st.write(f"Total rows in previous file: {results['total_rows_file1']}")

                with col1:
                    total_rows_file2 = add_commas(results['total_rows_file2'])  # dynamically get the value
                    st.markdown(
                        f"""
                                            <div class='column-box' style='text-align: left;'>
                                                <p style='text-align: center; color:black'><b>Rows in Current Month Data<b></p> 
                                                <p style='text-align: center; font-size: 150%;margin: -1rem'><b>{total_rows_file2}</b></p>
                                            </div>
                                            """,
                        unsafe_allow_html=True
                    )
                with col2:
                    total_rows_file1 = add_commas(testingdata[(testingdata['pred'] == True) & 
                                   (testingdata['residual_category'].isin(anomalies_category))].shape[0])
                    
                    #print("Number of Anomolous Data",total_rows_file1)
                    st.markdown(
                        f"""
                                            <div class='column-box' style='text-align: left;'>
                                                <p style='text-align: center; color:black'><b>Anomalies Detected <b></p>
                                                <p style='text-align: center; font-size: 150%;margin: -1rem'><b>{total_rows_file1}</b></p>

                                            </div>
                                            """,
                        unsafe_allow_html=True
                    )
                with col3:
                    new_brand=list(set(today_df['Brnd_Name'])-set(yesterday_df['Brnd_Name']))
                    new_brand=add_commas(len(new_brand))
                    #total_rows_file1 = add_commas(results['total_rows_file1'])  # dynamically get the value
                    st.markdown(
                        f"""
                        <div class='column-box' style='text-align: left;'>
                                <p style='text-align: center; color:black'><b> New Brand Count <b></p>
                                <p style='text-align: center; font-size: 150%;margin: -1rem'><b>{new_brand}</b></p>
                                
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col4:
                    #total_rows_file1 = add_commas(results['Changes_in_old_File'].shape[0]+results['deleted_records'].shape[0]+results['new_records'].shape[0])  # dynamically get the value
                    new_drug=len(set(today_df['Drug Type'])-set(yesterday_df['Drug Type']))
                    st.markdown(
                        f"""
                                            <div class='column-box' style='text-align: left;'>
                                                <p style='text-align: center; color:black'><b>New Drug Type Count<b></p>
                                                <p style='text-align: center; font-size: 150%;margin: -1rem'><b>{add_commas(new_drug)}</b></p>

                                            </div>
                                            """,
                        unsafe_allow_html=True
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                
                MonthSummary1,MonthSummary2 = st.columns(2)  # Adjust the width ratio
                with MonthSummary1:
                    st.markdown(f'<p class="title-font" style="color:black;margin-top:80px;" ><b>Last Month File Summary<b></p>', unsafe_allow_html=True)
                    MonthSummary(yesterday_df,brand_selected)
                with MonthSummary2:
                    st.markdown(f'<p class="title-font" style="color:black;margin-top:80px;" ><b>Current Month File Summary<b></p>', unsafe_allow_html=True)
                    MonthSummary(today_df,brand_selected) 
                
                st.markdown(f'''<p style="color:black;font-size: 15px;font-weight: bold;margin-top:80px;" >Anomalies ranges at {Dimension_selection} </p>''', unsafe_allow_html=True)    
                Stats_yesterday,Stats_today = st.columns(2)
                with Stats_yesterday:
                    st.markdown(f'''<p style="color:black;font-size: 15px;font-weight: bold;margin-top:20px;" >Last Month </p>''', unsafe_allow_html=True)
                    group_df = yesterday_df.groupby(Dimension_selection)[Metric_selection].agg(['mean', 'std', 'count', 'min', 'max']).reset_index()
                   
                    group_df.index=group_df.index+1    
                    st.dataframe(group_df,width=1500) 
                    
                with Stats_today:
                    st.markdown(f'''<p style="color:black;font-size: 15px;font-weight: bold;margin-top:20px;" >Current Month </p>''', unsafe_allow_html=True)
                    group_df = today_df.groupby(Dimension_selection)[Metric_selection].agg(['mean', 'std', 'count', 'min', 'max']).reset_index()
                   
                    group_df.index=group_df.index+1    
                    st.dataframe(group_df,width=1500) 
                
                
                BrandTrendGraph=st.columns(1)[0]
                with BrandTrendGraph:
                    st.markdown('<p style="color:black;font-size: 15px;font-weight: bold;margin-top:80px;" >Brand Trend Graph</p>', unsafe_allow_html=True)
                    Brand_Trend_Graph(yesterday_df,today_df, brand_selected, month_select, Metric_selection)           
                            
                threshold=st.columns(1)[0]
                with threshold:
                    st.markdown('<p style="color:black;font-size: 15px;font-weight: bold;margin-top:80px;" >Anomalies by Brand & State</p>', unsafe_allow_html=True)
                    threshold_df=testingdata.copy()
                    threshold_df['Month']=pd.to_datetime(threshold_df['Date']).dt.to_period('M')
                    threshold_df = threshold_df.groupby(['Brnd_Name', 'Prscrbr_State_Abrvtn', 'Month']).agg(
                        Rows=('Brnd_Name', 'size'),  # Total number of rows
                        Anomalies=('pred', lambda x: (x == 1).sum())  # Count of rows where pred=1
                    ).reset_index()
                    threshold_df.sort_values(by=['Month','Brnd_Name','Prscrbr_State_Abrvtn'], ascending=[True,True,True], inplace=True)
                    threshold_df=threshold_df.reset_index(drop=True)
                    threshold_df.index=threshold_df.index+1
                    st.dataframe(threshold_df,width=1500)
                
                
                col6_1 = st.columns(1)[0] 
                with col6_1:
                    st.markdown(f'''<p style="color:black;font-size: 15px;font-weight: bold;margin-top:80px;" >Anomalous data</p>''', unsafe_allow_html=True)
                    Anomolous_df=testingdata[testingdata['pred']==True]
                    Anomolous_df.drop(columns='pred',inplace=True)
                    Anomolous_df=Anomolous_df[Anomolous_df['residual_category'].isin(anomalies_category)]
                    Anomolous_df['Prscrbr_NPI']=Anomolous_df['Prscrbr_NPI'].astype(str)
                    category_order = ['Significant', 'Moderate', 'Minor', 'Non Anomalous']

                    # Convert 'residual_category' to an ordered categorical type
                    Anomolous_df['residual_category'] = pd.Categorical(Anomolous_df['residual_category'], 
                                                                    categories=category_order, 
                                                                    ordered=True)

                    # Sort the DataFrame based on the 'residual_category' ordinal values
                    Anomolous_df_sorted = Anomolous_df.sort_values(by='residual_category')
                    Anomolous_df_sorted=Anomolous_df_sorted.reset_index(drop=True)
                    Anomolous_df_sorted.index=Anomolous_df_sorted.index+1
                    st.dataframe(Anomolous_df_sorted)    
                
                col7,col7_1=st.columns(2)
                with col7_1:
                    
                    Anomalies_Brand(testingdata,Dimension_selection)  
                         
                with col7:
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
                            margin: 10px 0;
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
                            <p>Total No. of Anomalies Detected in Current Month Data Load</p>
                            <img  src='data:image/png;base64,{img2_base64}' style='width:80%;position:relative;
                            margin-top: 66px;'> 
                            <div class='legend-box'>
                                <div style="color:black" ><span class='green'></span><b>Non-Anomalous Records<b></div>
                                <div style="color:black" ><span class='red'></span><b>Anomalous Records<b></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                st.session_state.anomaly_button=False
                    
            if st.session_state.restatement_button :
                st.markdown('<p class="title-font" style="color:black;margin-top:10px" ><b>Overview of Data<b></p>', unsafe_allow_html=True)
                
                REKPI1, REKPI2, REKPI3, REKPI4 = st.columns(4)
                with REKPI1:
                    st.markdown(
                        f"""
                            <div class='column-box' style='text-align: left;'>
                                <p style='text-align: center; color:black'><b>Total Number of Rows <b></p> 
                                <p style='text-align: center; font-size: 150%;margin: -1rem'><b>{add_commas(today_df.shape[0])}</b></p>
                            </div>
                        """,
                        unsafe_allow_html=True
                    )
                with REKPI2:
                    st.markdown(
                        f"""
                            <div class='column-box' style='text-align: left;'>
                                <p style='text-align: center; color:black'><b>Modified Rows<b></p> 
                                <p style='text-align: center; font-size: 150%;margin: -1rem'><b>{add_commas(results['Changes_in_old_File'].shape[0])}</b></p>
                            </div>
                        """,
                        unsafe_allow_html=True
                    )
                with REKPI3:
                    st.markdown(
                        f"""
                            <div class='column-box' style='text-align: left;'>
                                <p style='text-align: center; color:black'><b>New Rows<b></p> 
                                <p style='text-align: center; font-size: 150%;margin: -1rem'><b>{add_commas(results['new_records'].shape[0])}</b></p>
                            </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                with REKPI4:
                    st.markdown(
                        f"""
                            <div class='column-box' style='text-align: left;'>
                                <p style='text-align: center; color:black'><b>Deleted Rows<b></p> 
                                <p style='text-align: center; font-size: 150%;margin: -1rem'><b>{add_commas(results['deleted_records'].shape[0])}</b></p>
                            </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                OVKPI1=st.columns(1)[0]
                with OVKPI1:
                    
                    st.markdown(f'<p class="title-font" style="color:black;margin-top:80px;" ><b>Structural Changes in the Current File - {min(today_df['Date']).date()} to {max(today_df['Date']).date()}   <b></p>', unsafe_allow_html=True)
                    structural_change(yesterday_df,today_df)
                    
                    # df_summary = pd.DataFrame({
                    #     "Row Number": [str(yesterday_df.shape[0])],
                    #     "Column Number": [str(yesterday_df.shape[1])],
                    #     "Numerical Columns": [
                    #         str(len(yesterday_df.select_dtypes(include=['int', 'float']).columns))
                    #     ],
                    #     "Categorical Columns": [
                    #         str(len(yesterday_df.select_dtypes(include=['object', 'category']).columns))
                    #     ],
                    #     "Date Columns": [
                    #         str(len(yesterday_df.select_dtypes(include=['datetime', 'datetime64']).columns))
                    #     ]
                    # }, index=["Last Data"])
                    #st.dataframe(df_summary,width=500)
                
                # with OVKPI2:
                #     st.markdown(f'<p class="title-font" style="color:black;margin-top:80px;" ><b>Current Month Data - {min(today_df['Date']).date()} to {max(today_df['Date']).date()}  <b></p>', unsafe_allow_html=True)
                #     df_summary = pd.DataFrame({
                #         "Row Number": [str(today_df.shape[0])],
                #         "Column Number": [str(today_df.shape[1])],
                #         "Numerical Columns": [
                #             str(len(today_df.select_dtypes(include=['int', 'float']).columns))
                #         ],
                #         "Categorical Columns": [
                #             str(len(today_df.select_dtypes(include=['object', 'category']).columns))
                #         ],
                #         "Date Columns": [
                #             str(len(today_df.select_dtypes(include=['datetime', 'datetime64']).columns))
                #         ]
                #     }, index=["Current Data"])
                #     st.dataframe(df_summary,width=500)
                
                
                st.markdown('<p class="title-font" style="color:black;margin-top:80px;" ><b>Preview of the Data <b></p>', unsafe_allow_html=True)
                
                
                with st.expander("Expand to see the preview of the data"):
                    st.markdown('<p class="title-font" style="color:black;margin-top:20px;" ><b>Current Month Data<b></p>', unsafe_allow_html=True)
                    today_df.index=today_df.index+1
                    preview_today=today_df.copy()
                    preview_today['Prscrbr_NPI']=preview_today['Prscrbr_NPI'].astype(int)
                    preview_today['Prscrbr_NPI']=preview_today['Prscrbr_NPI'].astype(str)
                    st.dataframe(preview_today.head(20))
                
                
                st.markdown('<p class="title-font" style="color:black;margin-top:80px" ><b>Dimension Level Distribution <b></p>', unsafe_allow_html=True)
                col1,col2=st.columns(2)
                with col1:
                    st.markdown('<p class="title-font" style="color:black;margin-top:20px;" ><b>Last Month Data<b></p>', unsafe_allow_html=True)
                    #plot_bar_frequency_graph(yesterday_df,'Prscrbr_Type')
                    plot_pie(yesterday_df, Dimension_selection)
                with col2:
                    st.markdown('<p class="title-font" style="color:black;margin-top:20px;" ><b>Current Month Data<b></p>', unsafe_allow_html=True)
                    plot_pie(today_df, Dimension_selection)             
                
                
                st.markdown('<p class="title-font" style="color:black;margin-top:80px;"><b>Summary of Restatement in Recent file <b></p>', unsafe_allow_html=True)
                Restatement_Summary=st.columns(1)[0]
                with Restatement_Summary:
                    print("For Now")
                    display_comparison_results(results, yesterday_df, today_df)
                    
                
                st.markdown('<p class="title-font" style="color:black;margin-top:80px;"><b>Modified Data<b></p>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<p class="title-font" style="color:black" >Last Month Data</p>', unsafe_allow_html=True)
                    st.dataframe(results['Changes_in_old_File'])

                with col2:
                    st.markdown('<p class="title-font" style="color:black" >Current Month Data</p>', unsafe_allow_html=True)
                    st.dataframe(results['Changes_in_recent_File_new'])                
                
                
                st.markdown('<p class="title-font" style="color:black;margin-top:80px;"><b>Deleted Rows from Recent file<b></p>', unsafe_allow_html=True)
                if results['deleted_records'].shape[0] ==0:
                    st.markdown('<p  style="color:black" ><i>There is no rows deleted from the Recent file</i></p>', unsafe_allow_html=True)
                else:
                    
                    st.dataframe(results['deleted_records'])
                    
                st.markdown('<p class="title-font" style="color:black;margin-top:80px;"><b>New Rows in Recent file<b></p>', unsafe_allow_html=True)
                if results['new_records'].shape[0] ==0:
                    st.markdown('<p  style="color:black" ><i>There is no new rows inserted in  the Recent file</i></p>', unsafe_allow_html=True)
                else:
                    st.dataframe(results['new_records'])
                
                st.markdown('<p class="title-font" style="color:black;margin-top:80px;"><b>New Data in Non Overlapping Period<b></p>', unsafe_allow_html=True)
                new_data_Non_Overlapped_Period=today_df[today_df['Date']>max(yesterday_df['Date'])]
                if new_data_Non_Overlapped_Period.shape[0]==0:
                    st.markdown('<p style="color:black" ><i>There is no new rows added in this recent file</i></p>', unsafe_allow_html=True)
                else:
                    new_data_Non_Overlapped_Period=new_data_Non_Overlapped_Period.reset_index(drop=True)
                    new_data_Non_Overlapped_Period.index=new_data_Non_Overlapped_Period.index+1
                    st.dataframe(new_data_Non_Overlapped_Period)
                
                st.session_state.restatement_button=False                    
        

if __name__ == "__main__":
    main()