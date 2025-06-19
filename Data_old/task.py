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
                'Requires': entry['keywords'][0].get('Requires', 'None')
            }
            data.append(entry_data)
    
    return pd.DataFrame(data) if data else pd.DataFrame(columns=['File Path', 'User Name', 'Question', 'Require Tool', 'Require Agent', 'Operation Type', 'Requires'])

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
        "user": [entry_data['User Name']]
    }
    
    with open(file_path, 'w') as f:
        json.dump(entry_json, f, indent=4)

# Main Page - Add New Entry
def main_page():
    st.title('Add New Entry')

    # Load data
    data = load_data()
    
    # User input form for new entry
    user_name = st.selectbox('User Name', ['Indrajeet', 'Jitendra'])
    question = st.text_area('Question')
    require_tool = st.radio('Require Tool', ['Single tool', 'Multiple tools'])
    require_agent = st.radio('Require Agent', ['Single agent', 'Multiple agents'])
    operation_type = st.radio('Operation Type', ['Basic', 'Advanced'])
    requires = st.multiselect('Requires', ['Documentation', 'Calculation', 'Analysis', 'None'])
    
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

    # View existing data
    if st.button('View All Entries'):
        st.write(data)

# Edit/Delete Page - Edit Existing Entries
def edit_page():
    st.title('Edit or Delete Entries')

    # Load data
    data = load_data()

    if data.empty:
        st.warning('No entries to display.')
        return

    # Sort entries numerically based on the suffix in filenames
    data['File Name'] = data['File Path'].apply(lambda x: os.path.basename(x))
    data['Numeric Suffix'] = data['File Name'].str.extract(r'(\d+)').astype(float)  # Extract number as float for sorting
    data = data.sort_values(by='Numeric Suffix').drop(columns='Numeric Suffix').reset_index(drop=True)

    # Initialize session state for index
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    # Total number of entries
    total_entries = len(data)

    # Get the current entry based on index
    current_index = st.session_state.current_index
    selected_entry = data.iloc[current_index]
    file_path = selected_entry['File Path']
    file_name = selected_entry['File Name']

    st.write(f"Editing Entry {current_index + 1} of {total_entries}")
    st.write(f"**File Name:** `{file_name}`")

    # Editing form
    question = st.text_area('Question', value=selected_entry['Question'], key='question')
    require_tool = st.radio('Require Tool', ['Single tool', 'Multiple tools'], 
                            index=['Single tool', 'Multiple tools'].index(selected_entry['Require Tool']),
                            key='require_tool')
    require_agent = st.radio('Require Agent', ['Single agent', 'Multiple agents'], 
                             index=['Single agent', 'Multiple agents'].index(selected_entry['Require Agent']),
                             key='require_agent')
    operation_type = st.radio('Operation Type', ['Basic', 'Advanced'], 
                              index=['Basic', 'Advanced'].index(selected_entry['Operation Type']),
                              key='operation_type')
    requires = st.multiselect('Requires', ['Documentation', 'Calculation', 'Analysis', 'None'], 
                              default=selected_entry['Requires'].split(', '),
                              key='requires')

    # Navigation buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button('Previous') and current_index > 0:
            st.session_state.current_index -= 1
            st.rerun()

    with col2:
        if st.button('Next') and current_index < total_entries - 1:
            st.session_state.current_index += 1
            st.rerun()

    # Save Changes button
    save_changes = st.button('Save Changes', key='save_changes')

    if save_changes:
        updated_entry = {
            'User Name': selected_entry['User Name'],
            'Question': st.session_state.question,
            'Require Tool': st.session_state.require_tool,
            'Require Agent': st.session_state.require_agent,
            'Operation Type': st.session_state.operation_type,
            'Requires': ', '.join(st.session_state.requires)
        }
        save_data(updated_entry, file_path)
        st.success('Entry updated successfully!')

    # Delete Entry button
    if st.button('Delete Entry', key='delete_entry'):
        os.remove(file_path)
        st.success('Entry deleted successfully!')
        st.session_state.current_index = max(0, current_index - 1)
        st.rerun()

# Streamlit page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Add Entry", "Edit Entries"])

if page == "Add Entry":
    main_page()
elif page == "Edit Entries":
    edit_page()
