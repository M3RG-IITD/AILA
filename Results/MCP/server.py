#server.py
from mcp.server.fastmcp import FastMCP
import getpass
import os
import functools
import operator
import glob
#import nanosurf #change
import time
import numpy as np
import matplotlib.pyplot as plt
from langchain_chroma import Chroma
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import tool
from NSFopen.read import read
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from scipy.optimize import curve_fit
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from typing import Sequence, TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

mcp = FastMCP("AILA")

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"


llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

db_new = Chroma(persist_directory="./aila_db", embedding_function=embeddings)

retriever_wo = db_new.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

Document_Retriever = create_retriever_tool(
    retriever_wo,
    "Document_Retriever",
    "This tool offers reference code specifically designed for "
    "managing an AFM (Atomic Force Microscope) machine, which is connected to a database. "
    "The tool also includes the AFM software manual for guidance."
    "However, it does not contain any code related to displaying/optimizing images."
    "Single query allowed. but multiple calls allowed",
)

@mcp.tool()
def Code_Executor(code: str) -> int:
    "Use this tool only to run Python code for operating an Atomic Force Microscope."
    "Use code from 'Document_Retriever' and correct it as needed. This code controls the AFM, so handle it with care."
    try:
        # Execute the code
        import pythoncom
        pythoncom.CoInitialize()
        exec(code)
        output='code executed successfully'
    except Exception as e:
        print("Error:", e) 
        output=e
    return output

@mcp.tool()
def Image_Analyzer(path: str = None, filename: str = None, dynamic_code: str = None, calculate_friction: bool = False, calculate_mean_roughness: bool = False, calculate_rms_roughness: bool = False):
    """
    Display and return the image data from the given path. If a filename is provided, return the image data
    from that specific file. If no filename is provided, return the image data from the latest image file
    in the directory. If dynamic_code is provided, it will be executed to process the image data. Don’t install any Python library or any softwere.
    
    Additionally, calculate the following if requested:
    - Average Friction
    - Mean Roughness
    - RMS Roughness

    Args:
    - path (str): The directory path to search for the latest file. Defaults to None.
    - filename (str): The specific image file to display. Defaults to None.
    - dynamic_code (str): A string containing Python code to process the image data. Defaults to None.
    - calculate_friction (bool): Whether to calculate average friction. Defaults to False.
    - calculate_mean_roughness (bool): Whether to calculate mean roughness. Defaults to False.
    - calculate_rms_roughness (bool): Whether to calculate RMS roughness. Defaults to False.

    Returns:
    - dict: A dictionary containing the status, image data, or an error message.
    """
    if path is None:
        path = os.getcwd()
    
    # Determine the file to display
    if filename:
        file_to_display = os.path.join(path, filename)
        if not os.path.isfile(file_to_display):
            print(f"File not found: {file_to_display}")
            return {"status": "Error", "message": "The specified file does not exist."}
    else:
        # Get the list of all files in the directory
        list_of_files = glob.glob(os.path.join(path, '*'))
        
        if not list_of_files:
            print("No files found in the specified directory.")
            return {"status": "Error", "message": "No files found in the directory."}
        
        # Find the latest file based on creation time
        file_to_display = max(list_of_files, key=os.path.getctime)
    
    print(f"File to display: {file_to_display}")

    try:
        # Read the file
        afm = read(file_to_display)
        
        # Extract data and parameters
        data = afm.data  # Raw data
        param = afm.param  # Parameters
        
        # Assuming 'Image', 'Forward', and 'Z-Axis' are keys in the data structure
        image_data = data['Image']['Forward']['Z-Axis']
        
        # If dynamic code is provided, execute it. image_data = data['Image']['Forward']['Z-Axis'] cange Forward to Backward if asked. Z-Axis to Deflection or Friction force if asked. 
        if dynamic_code:
            # Safely execute the dynamic code
            try:
                exec(dynamic_code)
                # After executing the dynamic code, `image_data` should be processed accordingly
                print("Dynamic code executed successfully.")
            except Exception as e:
                print(f"Error executing dynamic code: {e}")
                return {"status": "Error", "message": f"Error executing dynamic code: {str(e)}"}
        
        # Calculate Average Friction if requested
        if calculate_friction:
            friction = 0.5 * (data['Image']['Forward']['Friction force'] - data['Image']['Backward']['Friction force'])
            average_friction = np.mean(friction)
            print(f"Average Friction: {average_friction}")
        
        # Calculate Mean Roughness if requested
        if calculate_mean_roughness:
            z = data['Image']['Forward']['Z-Axis']
            z_mean = np.mean(z)
            absolute_differences = np.abs(z - z_mean)
            total_sum = np.sum(absolute_differences)
            M, N = z.shape
            mean_roughness = total_sum / (M * N)
            print(f"Mean Roughness: {mean_roughness}")
        
        # Calculate RMS Roughness if requested
        if calculate_rms_roughness:
            z = data['Image']['Forward']['Z-Axis']
            z_mean = np.mean(z)
            squared_differences = (z - z_mean) ** 2
            total_sum = np.sum(squared_differences)
            M, N = z.shape
            rms_roughness = np.sqrt(total_sum / (M * N))
            print(f"RMS Roughness: {rms_roughness}")
        
        # Return the image data along with status
        result = {"status": "Success", "message": f"Raw Image {file_to_display} processed successfully.", "image_data": image_data}
        
        # Include calculated metrics in the result if they were calculated
        if calculate_friction:
            result["average_friction"] = average_friction
        if calculate_mean_roughness:
            result["mean_roughness"] = mean_roughness
        if calculate_rms_roughness:
            result["rms_roughness"] = rms_roughness
        
        return result
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"status": "Error", "message": f"An error occurred: {str(e)}"}


def scan_image(PGain, IGain, DGain, file):
    spm = nanosurf.SPM()
    application = spm.application
    scan = application.Scan
    zcontrol = application.ZController
    
    application.SetGalleryHistoryFilenameMask(file)
    zcontrol.PGain = PGain
    zcontrol.IGain = IGain
    zcontrol.DGain = DGain
    scan.StartFrameUp()
    
    scanning = scan.IsScanning
    while scanning:
        print("Scanning in progress...")
        time.sleep(5)
        scanning = scan.IsScanning

    from NSFopen.read import read
    list_of_files = glob.glob(new_path+'/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    afm = read(latest_file)
    data = afm.data
    im_file_fw = data['Image']['Forward']['Z-Axis']
    im_file_bw = data['Image']['Backward']['Z-Axis']
    similarity_index, diff = ssim(im_file_bw, im_file_fw, full=True, data_range=im_file_bw.max() - im_file_bw.min())
    mse = mean_squared_error(im_file_bw, im_file_fw)
    del spm
    return similarity_index, mse

def corrected_image(image):
    def poly5d(xy, *params):
        x, y = xy
        return (params[0] + params[1]*x + params[2]*y + 
                params[3]*x**2 + params[4]*y**2 + 
                params[5]*x*y + params[6]*x**3 + params[7]*y**3 +
                params[8]*x**2*y + params[9]*x*y**2 + 
                params[10]*x**4 + params[11]*y**4 + 
                params[12]*x**3*y + params[13]*x*y**3 +
                params[14]*x**2*y**2)

    x = np.arange(image.shape[1])
    y = np.arange(image.shape[0])
    x, y = np.meshgrid(x, y)
    x = x.flatten()
    y = y.flatten()
    image_flat = image.flatten()
    params, _ = curve_fit(poly5d, (x, y), image_flat, p0=np.zeros(15))
    baseline = poly5d((x, y), *params).reshape(image.shape)
    corrected_image = image - baseline
    return corrected_image

def scan_image_poly(PGain, IGain, DGain, file):
    spm = nanosurf.SPM()
    application = spm.application
    scan = application.Scan
    zcontrol = application.ZController
    
    application.SetGalleryHistoryFilenameMask(file)
    zcontrol.PGain = PGain
    zcontrol.IGain = IGain
    zcontrol.DGain = DGain
    scan.StartFrameUp()
    
    scanning = scan.IsScanning
    while scanning:
        print("Scanning in progress...")
        time.sleep(5)
        scanning = scan.IsScanning

    from NSFopen.read import read
    list_of_files = glob.glob(new_path+'/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    afm = read(latest_file)
    data = afm.data
    im_file_fw = corrected_image(data['Image']['Forward']['Z-Axis'])
    im_file_bw = corrected_image(data['Image']['Backward']['Z-Axis'])
    similarity_index, diff = ssim(im_file_bw, im_file_fw, full=True, data_range=im_file_bw.max() - im_file_bw.min())
    mse = mean_squared_error(im_file_bw, im_file_fw)
    del spm
    return similarity_index, mse

class MyProblem(ElementwiseProblem):
    def __init__(self, baseline=True):
        super().__init__(n_var=3,
                         n_obj=1,
                         xl=np.array([0, 500, 0]),
                         xu=np.array([500, 9000, 100]))
        self.baseline = baseline

    def _evaluate(self, x, out, *args, **kwargs):
        if self.baseline:
            scan_outputs = scan_image_poly(x[0], x[1], x[2], f"scan_{x[0]}_{x[1]}_{x[2]}_")
        else:
            scan_outputs = scan_image(x[0], x[1], x[2], f"scan_{x[0]}_{x[1]}_{x[2]}_")

        mse = scan_outputs[1]
        f1 = (1 - mse) * 10000
        out["F"] = [f1]

# Save the current working directory
original_path = os.getcwd()

# Specify the new path you want to temporarily switch to
new_path = '/Users/Admin/AppData/Local/Nanosurf/Nanosurf CX/AI-Optimization'

@mcp.tool()
def Image_optimizer(baseline: bool) -> str:
    """This tool optimizes the parameters (P/I/D gains) based on baseline correction
    settings to provide the best solution for image clarity. Use this tool if the image 
    appears blurry or unclear and you want to enhance its sharpness."""

    try:
        os.chdir(new_path)
        print(f"Current working directory: {os.getcwd()}")
        
        # Your code that needs to be executed in the new directory goes here
        import pythoncom
        pythoncom.CoInitialize()
    
        list_of_files = glob.glob(new_path+'/*') 
        latest_file = max(list_of_files, key=os.path.getctime)
        print(latest_file)
    
    
        problem = MyProblem(baseline=baseline)
    
        termination = get_termination("n_gen", 2)
        algorithm = GA(pop_size=2, eliminate_duplicates=True)
    
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       verbose=True)

    finally:
        # Restore the original working directory
        os.chdir(original_path)
        print(f"Returned to original working directory: {os.getcwd()}")

    return "Best solution found: \n[Pgain Igain Dgain] = %s\n[Error] = %s" % (res.X, res.F)



if __name__ == "__main__":
    mcp.run(transport="stdio")