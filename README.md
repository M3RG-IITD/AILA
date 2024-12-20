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
AILA requires Python and the Streamlit library to run. Please ensure Python is installed on your system before proceeding. You can download it from [python.org](https://www.python.org/).

---

## **Installation Instructions**

### **Windows**

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/imandal98/AFMBench.git
   cd AFMBench
   ```

2. **Install Streamlit:**
   Install the Streamlit library using pip:
   ```bash
   pip install streamlit
   ```

3. **Run the Application:**
   - Double-click `run_app.bat` to launch the main application.
   - To view the questions, double-click `run_bat`.

### **Mac**

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/imandal98/AFMBench.git
   cd AFMBench
   ```

2. **Install Streamlit:**
   Install the Streamlit library using pip:
   ```bash
   pip install streamlit
   ```

3. **Navigate to the Data Folder:**
   ```bash
   cd Data
   ```

4. **Run the Application:**
   - To view the questions, execute the following command:
     ```bash
     streamlit run afm_data_json.py
     ```
   - To launch the main application, navigate to the results folder and run the Streamlit script:
     ```bash
     streamlit run afm_data_json.py
     ```

---

## **Benchmark Data and Results**

The benchmark data and results for AILA are available in the `Results` folder. Navigate to the following GitHub repository to access them:

[AILA Results Folder](https://github.com/M3RG-IITD/AILA/tree/main/Results/)

- **Data Folder:** Contains all input data files used for benchmarking.
- **Results Folder:** Contains the output results of the AILA framework.

---

## **Support**
If you encounter any issues or have questions, please open an issue on the [GitHub Repository](https://github.com/imandal98/AFMBench) or contact the project maintainer.

---

Thank you for using AILA! We hope it enhances your laboratory workflows.

