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
        return pd.DataFrame(columns=['User Name', 'Question', 'Require Tool', 'Require Agent', 'Operation Type', 'Requires', 'Result GPT-4', 'Result GPT-3.5'])

# Save data to Excel
def save_data(data):
    data.to_excel(file_path, index=False)

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
    
    data = load_data()
    
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
            st.error("All fields are required")

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
    entry_index = data[data['Question'] == question].index[0]
    
    # Fetch existing results if available
    existing_gpt4_results = eval(data.at[entry_index, 'Result GPT-4']) if pd.notna(data.at[entry_index, 'Result GPT-4']) else []
    existing_gpt35_results = eval(data.at[entry_index, 'Result GPT-3.5']) if pd.notna(data.at[entry_index, 'Result GPT-3.5']) else []
    
  # Function to add multiple results
    def add_results(model_name, existing_results):
        results = []
        for i in range(1, 4):
            st.subheader(f'Result {i} for {model_name}')

            # Set default values if existing results are available
            time = st.number_input(f'Time Required for {model_name} (seconds) - Entry {i}', min_value=0.0, step=0.1, value=existing_results[i-1]['Time'] if i <= len(existing_results) else 0.0)
            
            correct = st.radio(f'Final Answer is Correct for {model_name} - Entry {i}', ['Yes', 'No'], index=0 if (i > len(existing_results) or existing_results[i-1]['Correct'] == 'No') else 1, key=f'correct_{model_name}_{i}')

            # Handle default for agent with safe check
            existing_agents = existing_results[i-1]['Agent'].split(', ') if i <= len(existing_results) else []
            available_agents = ['AFM Operation Handler', 'Data Handler', 'None']
            agent = st.multiselect(f'Agent Name for {model_name} - Entry {i}', available_agents, default=[a for a in existing_agents if a in available_agents], key=f'agent_{model_name}_{i}')
            
            # Handle default for tool with safe check
            existing_tools = existing_results[i-1]['Tool'].split(', ') if i <= len(existing_results) else []
            available_tools = ['Document retriever', 'Code executor', 'Image optimizer', 'Image analyzer', 'None']
            tool = st.multiselect(f'Tool Name for {model_name} - Entry {i}', available_tools, default=[t for t in existing_tools if t in available_tools], key=f'tool_{model_name}_{i}')

            result = {
                'Time': time,
                'Correct': correct,
                'Agent': ', '.join(agent),
                'Tool': ', '.join(tool)
            }
            results.append(result)
        return results

    gpt4_results = add_results('GPT-4', existing_gpt4_results)
    gpt35_results = add_results('GPT-3.5', existing_gpt35_results)

    if st.button('Save Results'):
        # Update the selected entry with results
        data.at[entry_index, 'Result GPT-4'] = str(gpt4_results)
        data.at[entry_index, 'Result GPT-3.5'] = str(gpt35_results)
        
        save_data(data)
        st.success('Results saved successfully!')

# New Page - Show All Updated Results
def results_page():
    st.title('Updated Results for GPT-4 and GPT-3.5')

    data = load_data()

    if data.empty:
        st.warning('No results available to display.')
        return

    # Display the results in a table format
    results_columns = ['User Name', 'Question', 'Result GPT-4', 'Result GPT-3.5']
    st.write(data[results_columns])

# Edit/Delete Page - Edit Existing Entries
def edit_page():
    st.title('Edit or Delete Entries')
    
    data = load_data()
    
    if data.empty:
        st.warning('No entries to display.')
        return
    
    index = st.number_input('Entry Index', min_value=0, max_value=len(data)-1, step=1)
    
    st.write(f'**Current Entry {index}:**')
    st.write(data.iloc[index])
    
    question = st.text_area('Question', value=data.iloc[index]['Question'])
    require_tool = st.radio('Require Tool', ['Single tool', 'Multiple tools'], 
                            index=['Single tool', 'Multiple tools'].index(data.iloc[index]['Require Tool']))
    require_agent = st.radio('Require Agent', ['Single agent', 'Multiple agents'], 
                            index=['Single agent', 'Multiple agents'].index(data.iloc[index]['Require Agent']))
    operation_type = st.radio('Operation Type', ['Basic', 'Advanced'], 
                              index=['Basic', 'Advanced'].index(data.iloc[index]['Operation Type']))
    requires = st.multiselect('Requires', ['Documentation', 'Calculation', 'Reasoning', 'None'], 
                              default=data.iloc[index]['Requires'].split(', '))

    if st.button('Save Changes'):
        data.at[index, 'Question'] = question
        data.at[index, 'Require Tool'] = require_tool
        data.at[index, 'Require Agent'] = require_agent
        data.at[index, 'Operation Type'] = operation_type
        data.at[index, 'Requires'] = ', '.join(requires)
        save_data(data)
        st.success('Entry updated successfully!')
    
    if st.button('Delete Entry'):
        data = data.drop(index)
        save_data(data)
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
