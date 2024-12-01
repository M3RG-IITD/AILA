import streamlit as st
import pandas as pd
import os

# File path for the Excel database
file_path = 'database_ind.xlsx'

# Load Excel database
def load_data():
    if os.path.exists(file_path):
        return pd.read_excel(file_path)
    else:
        return pd.DataFrame(columns=['Question', 'Require Tool', 'Operation Type', 'Requires'])

# Save data to Excel
def save_data(data):
    data.to_excel(file_path, index=False)

# Main Page - Add New Entry
def main_page():
    st.title('Add New Entry')

    user_name=st.selectbox('User Name', ['Indrajeet', 'Jitendra'])
    question = st.text_area('Question')
    require_tool = st.radio('Require Tool', ['Single tool', 'Multiple tools'])
    require_agent = st.radio('Require Agent', ['Single agent', 'Multiple agents'])
    operation_type = st.radio('Operation Type', ['Basic', 'Advanced'])
    requires = st.multiselect('Requires', ['Documentation', 'Calculation', 'Reasoning', 'None'])
    
    if len(requires) == 0:
        st.error("Please select at least one 'Requires' option.")
    # Load existing data
    data = load_data()
    
    # Add new entry
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
            data = pd.concat([data, pd.DataFrame([new_entry])], ignore_index=True)
            save_data(data)
            st.success('Entry added successfully!')
        else:
            st.error("All field are required")

    if st.button('View All Entries'):
        st.write(data)

    # Filter by user name
    users = data['User Name'].unique()
    selected_user = st.selectbox('Select User to View Entries', users)
    if selected_user:
        filtered_data = data[data['User Name'] == selected_user]
        st.write(filtered_data)


# Edit/Delete Page - Edit Existing Entries
def edit_page():
    st.title('Edit or Delete Entries')
    
    data = load_data()
    
    if data.empty:
        st.warning('No entries to display.')
        return
    
    # Display a selector for choosing which entry to edit
    index = st.number_input('Entry Index', min_value=0, max_value=len(data)-1, step=1)
    
    # Display current entry data
    st.write(f'**Current Entry {index}:**')
    st.write(data.iloc[index])
    
    # Edit the current entry
    question = st.text_area('Question', value=data.iloc[index]['Question'])
    require_tool = st.radio('Require Tool', ['Single tool', 'Multiple tools'], 
                            index=['Single tool', 'Multiple tools'].index(data.iloc[index]['Require Tool']))
    require_agent = st.radio('Require Agent', ['Single agent', 'Multiple agents'], 
                            index=['Single agent', 'Multiple agents'].index(data.iloc[index]['Require Agent']))
    operation_type = st.radio('Operation Type', ['Basic', 'Advanced'], 
                              index=['Basic', 'Advanced'].index(data.iloc[index]['Operation Type']))
    requires = st.multiselect('Requires', ['Documentation', 'Calculation', 'Reasoning', 'None'], 
                              default=data.iloc[index]['Requires'].split(', '))

    # Save edited entry
    if st.button('Save Changes'):
        data.at[index, 'Question'] = question
        data.at[index, 'Require Tool'] = require_tool
        data.at[index, 'Require Agent'] = require_agent
        data.at[index, 'Operation Type'] = operation_type
        data.at[index, 'Requires'] = ', '.join(requires)
        save_data(data)
        st.success('Entry updated successfully!')
    
    # Delete entry
    if st.button('Delete Entry'):
        data = data.drop(index)
        save_data(data)
        st.success('Entry deleted successfully!')

# Streamlit page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Add Entry", "Edit Entries"])

if page == "Add Entry":
    main_page()
elif page == "Edit Entries":
    edit_page()
