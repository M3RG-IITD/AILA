import streamlit as st
import pandas as pd
import json
import os
from glob import glob

# Directory where JSON files are stored
json_dir = './afm_qs/'

# Load data from JSON files
def load_data():
    data = []
    json_files = glob(os.path.join(json_dir, 'question_*.json'))

    for json_file in json_files:
        with open(json_file, 'r') as f:
            entry = json.load(f)
            entry_data = {
                'File Path': json_file,  # Store file path for later updates
                'User Name': entry['user'][0],
                'Question': entry['question'][0]['input'],
                'Require Tool': entry['keywords'][0]['Require Tool'],
                'Require Agent': entry['keywords'][0]['Require Agent'],
                'Operation Type': entry['keywords'][0]['Operation Type'],
                'Requires': entry['keywords'][0].get('Requires', 'None'),
                'Result GPT-4': entry['result'][0].get('GPT-4', 'NaN'),
                'Result GPT-3.5': entry['result'][0].get('GPT-3.5', 'NaN')
            }
            data.append(entry_data)
    
    return pd.DataFrame(data) if data else pd.DataFrame(columns=['File Path', 'User Name', 'Question', 'Require Tool', 'Require Agent', 'Operation Type', 'Requires', 'Result GPT-4', 'Result GPT-3.5'])

# Save data to a JSON file (create or overwrite file for an entry)
def save_data(entry_data, file_path):
    entry_json = {
        "question": [{"input": entry_data['Question']}],
        "keywords": [{
            "Operation Type": entry_data['Operation Type'],
            "Require Tool": entry_data['Require Tool'],
            "Require Agent": entry_data['Require Agent'],
            "Requires": entry_data['Requires']
        }],
        "result": [{
            "GPT-4": entry_data.get('Result GPT-4', 'NaN'),
            "GPT-3.5": entry_data.get('Result GPT-3.5', 'NaN')
        }],
        "user": [entry_data['User Name']]
    }
    
    with open(file_path, 'w') as f:
        json.dump(entry_json, f, indent=4)

# Main Page - Add New Entry
def main_page():
    st.title('Add New Entry')

    user_name = st.selectbox('User Name', ['Indrajeet', 'Jitendra'])
    question = st.text_area('Question')
    require_tool = st.radio('Require Tool', ['Single tool', 'Multiple tools'])
    require_agent = st.radio('Require Agent', ['Single agent', 'Multiple agents'])
    operation_type = st.radio('Operation Type', ['Basic', 'Advanced'])
    requires = st.multiselect('Requires', ['Documentation', 'Calculation', 'Reasoning', 'None'])
    
    if len(requires) == 0:
        st.error("Please select at least one 'Requires' option.")
    
    if st.button('Add Entry'):
        if user_name and question and len(requires) > 0:
            new_entry = {
                'User Name': user_name,
                'Question': question,
                'Require Tool': require_tool,
                'Require Agent': require_agent,
                'Operation Type': operation_type,
                'Requires': ', '.join(requires)
            }
            save_data(new_entry, None)  # Save as a new JSON file
            st.success('Entry added successfully!')
        else:
            st.error("All fields are required")

    data = load_data()
    if st.button('View All Entries'):
        st.write(data)

    users = data['User Name'].unique()
    selected_user = st.selectbox('Select User to View Entries', users)
    if selected_user:
        filtered_data = data[data['User Name'] == selected_user]
        st.write(filtered_data)

# New Page - Add Results for GPT-4 and GPT-3.5
def add_results_page():
    st.title('Add Results for GPT-4 and GPT-3.5')

    data = load_data()

    if data.empty:
        st.warning('No entries available to add results.')
        return
    
    # Select entry by question
    question = st.selectbox('Select Question', data['Question'].unique())
    selected_entry = data[data['Question'] == question].iloc[0]
    file_path = selected_entry['File Path']
    
    # Fetch existing results if available
    existing_gpt4_results = selected_entry['Result GPT-4'] if pd.notna(selected_entry['Result GPT-4']) else 'NaN'
    existing_gpt35_results = selected_entry['Result GPT-3.5'] if pd.notna(selected_entry['Result GPT-3.5']) else 'NaN'
    
    # Input fields for new results
    gpt4_result = st.text_area('Result for GPT-4', value=str(existing_gpt4_results))
    gpt35_result = st.text_area('Result for GPT-3.5', value=str(existing_gpt35_results))

    if st.button('Save Results'):
        selected_entry['Result GPT-4'] = gpt4_result
        selected_entry['Result GPT-3.5'] = gpt35_result
        save_data(selected_entry, file_path)
        st.success('Results saved successfully!')

# View Results Page
def results_page():
    st.title('Updated Results for GPT-4 and GPT-3.5')

    data = load_data()

    if data.empty:
        st.warning('No results available to display.')
        return

    results_columns = ['User Name', 'Question', 'Result GPT-4', 'Result GPT-3.5']
    st.write(data[results_columns])

# Edit/Delete Page - Edit Existing Entries
# Edit/Delete Page - Edit Existing Entries
# Edit/Delete Page - Edit Existing Entries
# Edit/Delete Page - Edit Existing Entries
def edit_page():
    st.title('Edit or Delete Entries')

    data = load_data()

    if data.empty:
        st.warning('No entries to display.')
        return

    # Sort entries numerically based on the suffix in filenames
    data['File Name'] = data['File Path'].apply(lambda x: os.path.basename(x))
    data['Numeric Suffix'] = data['File Name'].str.extract(r'(\d+)').astype(float)  # Extract number as float for sorting
    data = data.sort_values(by='Numeric Suffix').drop(columns='Numeric Suffix')

    # Select a file name from the sorted list
    file_names = data['File Name'].tolist()
    selected_file_name = st.selectbox('Choose File', file_names)

    # Get the selected entry based on the selected file name
    selected_entry = data[data['File Name'] == selected_file_name].iloc[0]
    file_path = selected_entry['File Path']

    # Display current entry details with editing options
    st.write(f"File Name: {os.path.basename(file_path)}")
    st.write(f"User: {selected_entry['User Name']}")

    question = st.text_area('Question', value=selected_entry['Question'])
    require_tool = st.radio('Require Tool', ['Single tool', 'Multiple tools'], 
                            index=['Single tool', 'Multiple tools'].index(selected_entry['Require Tool']))
    require_agent = st.radio('Require Agent', ['Single agent', 'Multiple agents'], 
                             index=['Single agent', 'Multiple agents'].index(selected_entry['Require Agent']))
    operation_type = st.radio('Operation Type', ['Basic', 'Advanced'], 
                              index=['Basic', 'Advanced'].index(selected_entry['Operation Type']))
    requires = st.multiselect('Requires', ['Documentation', 'Calculation', 'Reasoning', 'None'], 
                              default=selected_entry['Requires'].split(', '))

    # Save Changes button
    if st.button('Save Changes'):
        selected_entry['Question'] = question
        selected_entry['Require Tool'] = require_tool
        selected_entry['Require Agent'] = require_agent
        selected_entry['Operation Type'] = operation_type
        selected_entry['Requires'] = ', '.join(requires)
        save_data(selected_entry, file_path)
        st.success('Entry updated successfully!')
    
    # Delete Entry button
    if st.button('Delete Entry'):
        os.remove(file_path)
        st.success('Entry deleted successfully!')

# Streamlit page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Add Entry", "Edit Entries", "Add Results", "View Results"])

if page == "Add Entry":
    main_page()
elif page == "Edit Entries":
    edit_page()
elif page == "Add Results":
    add_results_page()
elif page == "View Results":
    results_page()
