import streamlit as st
import subprocess

# Function to modify AILA.py or AILA.py_3.5 and run it
def modify_and_run_script(user_input, output_file, model):
    # Select the script based on the model

    if model == 'GPT-4o':
        script_path = 'AILA_4.0.py'
    elif model == 'llama-3.3-70b':
        script_path = 'AILA_llama_3.3.py'
    elif model == 'Deepseek':
        script_path = 'AILA_deepseek.py'
    elif model == 'Claude-3-sonnet':
        script_path = 'AILA_claude-3-son.py'
    elif model == 'Mixtral':
        script_path = 'AILA_mixtral.py'
    else:
        script_path = 'AILA_3.5.py'
    
    # Read the original script
    with open(script_path, 'r') as file:
        lines = file.readlines()

    # Modify the specific line (assuming you want to modify the input line)
    for i, line in enumerate(lines):
        if "inputs = {" in line:
            # Modify the line with the new user input
            lines[i] = f'inputs = {{"messages": [("user", "{user_input}")]}}\n'
            break

    # Write the modified script back to the selected AILA script
    with open(script_path, 'w') as file:
        file.writelines(lines)

    # Run the script and redirect output
    command = f'python {script_path} > {output_file}'
    try:
        subprocess.run(command, shell=True, check=True)
        return f"Script executed successfully. Output written to {output_file}."
    except subprocess.CalledProcessError as e:
        return f"An error occurred: {e}"

# Streamlit UI
st.title("AILA (Artificially Intelligent Laboratory Assistant)")

# User input fields
user_input = st.text_area("Enter your message:")
output_file = st.text_input("Enter the output filename (e.g., output.txt):", "output.txt")

# Dropdown for model selection
model = st.selectbox("Choose the model:", ("GPT-4o", "GPT-3.5 Turbo", "llama-3.3-70b", "Deepseek","Claude-3-sonnet", "Mixtral"))

# Run button
if st.button("Start"):
    if user_input and output_file:
        result = modify_and_run_script(user_input, output_file, model)
        st.success(result)
    else:
        st.error("Please fill in both fields.")
