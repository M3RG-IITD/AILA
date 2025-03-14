# AILA: Artificially Intelligent Laboratory Assistant

AILA (Artificially Intelligent Laboratory Assistant) is a streamlined framework designed for laboratory automation and assistance. This guide provides details on how to set up and run AILA on both Windows and macOS platforms, and where to find the benchmark data and results.

---

## **Table of Contents**
1. [Getting Started](#getting-started)
2. [Installation Instructions](#installation-instructions)
   - [Windows](#windows)
   - [Mac](#mac)
3. [Benchmark Data and Results](#benchmark-data-and-results)
4. [Support](#support)

---

## **Getting Started**
AILA requires Python and the Streamlit library to run. Please ensure Python 3.10 or higher is installed on your system before proceeding. You can download it from [python.org](https://www.python.org/). It is highly recommended to use a new virtual environment.

---

## **Installation Instructions**

### **Clone the Repository**
```bash
git clone https://github.com/M3RG-IITD/AILA.git
cd AILA
```

### **Install Required Packages**
Run the following command to install all necessary dependencies:
```bash
pip install streamlit matplotlib langchain_chroma langchain NSFopen scikit-image pymoo langchain_openai langgraph
```

### **Run AFM Bench Tasks**
1. Navigate to the `Data` folder:
   ```bash
   cd Data
   ```
2. Run the AFM Bench Tasks using Streamlit:
   ```bash
   streamlit run task.py
   ```
   Wait a few minutes as it will take some time to load all the files.

### **Run AILA**
1. Navigate back to the AILA main directory:
   ```bash
   cd ..
   ```  
2. Enter the "app" folder:
   ```bash
   cd app
   ```
3. Run the AILA application:
   ```bash
   streamlit run app.py
   ```
4. **LLM Model Setup:**
   - Open the corresponding script in the "app" folder for the desired LLM model:
     - **"AILA_claude-3-son.py"** for Claude
     - **"AILA_3.5.py"** for GPT-3.5
     - **"AILA_4.0.py"** for GPT-4.0
     - **"AILA_llama_3.3.py"** for Llama
   - Search for `YOUR_API_KEY` in the script and paste your OpenAI API key into the embedding model and corresponding LLM model sections.

---

## **Benchmark Data and Results**
The benchmark data and results for AILA are available in the `Data` and `Results` folders. Navigate to the following GitHub repository to access them:

- **[Data Folder](https://github.com/M3RG-IITD/AILA/tree/main/Data/afm_qs/):** Contains all input data files used for benchmarking.
- **[Results Folder](https://github.com/M3RG-IITD/AILA/tree/main/Results/):** Contains the output results of the AILA framework.

---

## **Support**
If you encounter any issues or have questions, please open an issue on the [GitHub Repository](https://github.com/M3RG-IITD/AILA) or contact the project maintainer.

---

Thank you for using AILA! We hope it enhances your laboratory workflows.

