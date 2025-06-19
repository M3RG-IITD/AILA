#!/usr/bin/env python
# coding: utf-8

# # AILA (Artificially Intelligent laboratory Assitant)
# 
# AILA (Artificially Intelligent Laboratory Assistant) is an advanced multi-agent system developed by the NT(M3)RG lab—a collaboration between the Multiphysics & Multiscale Mechanics Research Group (M3RG) and the Nanoscale Tribology, Mechanics & Microscopy of Materials (NTM3) Group—at the Indian Institute of Technology Delhi. AILA autonomously manages Atomic Force Microscope (AFM) imaging and analysis using a sophisticated combination of large language models (LLMs) and specially designed tools. This system is compatible with various AFM instruments, including DriveAFM by Nanosurf, offering seamless image optimization and capturing through a Python API. AILA represents a significant leap in laboratory automation, enabling precise and efficient AFM operations with minimal human intervention.

# In[1]:
from aila_image_process import *
import getpass
import os
import functools
import operator
import glob
import nanosurf #change
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
from typing import Dict
from typing import Sequence, TypedDict, Annotated
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph, START
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"


llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)


# ## Tools

# ### Document retriever tool

# In[3]:

db_new = Chroma(persist_directory="./aila_db", embedding_function=embeddings)

retriever_wo = db_new.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)


# In[4]:

Document_Retriever = create_retriever_tool(
    retriever_wo,
    "Document_Retriever",
    "This tool offers reference code specifically designed for "
    "managing an AFM (Atomic Force Microscope) machine, which is connected to a database. "
    "The tool also includes the AFM software manual for guidance."
    "However, it does not contain any code related to displaying/optimizing images."
    "Single query allowed. but multiple calls allowed",
)


# ### Code executor tool

# In[6]:


@tool
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


# ### Image analyzer tool

# In[7]:

@tool
def Image_Analyzer(path: str = None, filename: str = None, dynamic_code: str = None,
                   calculate_friction: bool = False, calculate_mean_roughness: bool = False,
                   calculate_rms_roughness: bool = False):
    """
    Analyze and process AFM image data from a .nid file.

    The function loads image data from the specified directory or file. If dynamic_code is provided,
    it is executed with access to image data, AFM parameters, and NumPy. Any variables defined in the
    dynamic code (e.g., custom calculations or modified image data) are captured and returned.

    Additionally, it can calculate:
    - Average Friction (based on forward and backward scans)
    - Mean Roughness (Ra)
    - RMS Roughness (Rq)

    Args:
    - path (str): Directory to search for .nid files. Defaults to current working directory.
    - filename (str): Specific .nid file to load. If None, the latest .nid file in the directory is used.
    - dynamic_code (str): Python code as a string to dynamically process the image data. The code can access:
        - image_data (initially: data['Image']['Forward']['Z-Axis'])
        - data (raw AFM data)
        - param (AFM scan parameters)
        - np (NumPy)
        You may also reassign image_data or define new variables.
    - calculate_friction (bool): Whether to calculate average friction force.
    - calculate_mean_roughness (bool): Whether to calculate mean surface roughness (Ra).
    - calculate_rms_roughness (bool): Whether to calculate RMS surface roughness (Rq).

    Returns:
    - dict: {
        "status": "Success" or "Error",
        "message": Informational message,
        "image_data": Final image data used (possibly modified by dynamic code),
        "average_friction": (if requested),
        "mean_roughness": (if requested),
        "rms_roughness": (if requested),
        "dynamic_output": Dictionary of all variables created in dynamic code
      }
    """
    if path is None:
        path = os.getcwd()

    # Determine file to display
    if filename:
        file_to_display = os.path.join(path, filename)
        if not os.path.isfile(file_to_display):
            return {"status": "Error", "message": f"The specified file does not exist: {file_to_display}"}
    else:
        list_of_files = glob.glob(os.path.join(path, '*.nid'))
        if not list_of_files:
            return {"status": "Error", "message": "No .nid files found in the directory."}
        file_to_display = max(list_of_files, key=os.path.getctime)

    try:
        afm = read(file_to_display)
        data = afm.data
        param = afm.param
        image_data = data['Image']['Forward']['Z-Axis']

        # Prepare local execution context
        exec_locals = {
            "data": data,
            "param": param,
            "image_data": image_data,
            "np": np
        }
        dynamic_output = {}

        if dynamic_code:
            try:
                exec(dynamic_code, {}, exec_locals)
                image_data = exec_locals.get("image_data", image_data)
                dynamic_output = {
                    key: value for key, value in exec_locals.items()
                    if key not in ["data", "param", "np"] and not key.startswith("__")
                }
            except Exception as e:
                return {"status": "Error", "message": f"Error executing dynamic code: {str(e)}"}

        result = {
            "status": "Success",
            "message": f"Raw Image {file_to_display} processed successfully.",
            "image_data": image_data,
            "dynamic_output": dynamic_output
        }

        if calculate_friction:
            friction = 0.5 * (data['Image']['Forward']['Friction force'] - data['Image']['Backward']['Friction force'])
            average_friction = np.mean(friction)
            result["average_friction"] = average_friction

        if calculate_mean_roughness:
            z = data['Image']['Forward']['Z-Axis']
            z_mean = np.mean(z)
            mean_roughness = np.mean(np.abs(z - z_mean))
            result["mean_roughness"] = mean_roughness

        if calculate_rms_roughness:
            z = data['Image']['Forward']['Z-Axis']
            z_mean = np.mean(z)
            rms_roughness = np.sqrt(np.mean((z - z_mean) ** 2))
            result["rms_roughness"] = rms_roughness

        return result

    except Exception as e:
        return {"status": "Error", "message": f"An error occurred: {str(e)}"}
        
# ### Image optimizer tool

# In[8]:

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


# In[9]:


# Save the current working directory
original_path = os.getcwd()

# Specify the new path you want to temporarily switch to
new_path = '/Users/Admin/Desktop/Automation/AILA2/AILA/Results/app/scan'

    
@tool
def Image_optimizer(baseline: bool) -> str:
    """
    Optimize image clarity by tuning P/I/D control parameters using a genetic algorithm.

    This tool enhances image sharpness by finding optimal P, I, and D gain values.
    It is particularly effective for images that appear blurry or contain small, 
    fine features.

    Args:
        baseline (bool): Set to True if the image contains very small or subtle features for example feature size less than 1000 nm.
                         Enables baseline correction to improve sensitivity during optimization.

    Returns:
        str: A formatted string showing the best P, I, D gains and the corresponding error value.
    """

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
    
        termination = get_termination("n_gen", 5)
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


# ## Agents

# In[10]:
@tool
def visualize_grain_boxes(image_path: str=None) -> list:
    """
    Detects grains in the input image and generates bounding boxes around each one.

    Each detected grain is assigned a unique index (starting from 1), which can be
    used to reference and process specific grains in subsequent steps (e.g., scanning).

    Args:
         image_path (str, optional): Path to the input image file. If None, uses the latest saved .nid file.

    Returns:
        list: List of bounding boxes as (index, x1, y1, x2, y2), where
              (x1, y1) is the bottom-left and (x2, y2) is the top-right corner.
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend for saving
    
    if image_path is None:
        # Search for the latest .nid file
        nid_files = glob.glob("*.nid")
        if not nid_files:
            raise FileNotFoundError("No .nid image files found in the current directory.")
        image_path = max(nid_files, key=os.path.getmtime)

    indexed_boxes, extents, Z_flat2, labeled = image_process(image_path)
    
    fig, ax = plt.subplots()
    clip = [np.percentile(Z_flat2, percent) for percent in [3, 91]]
    ax.imshow(Z_flat2, cmap='afmhot', origin='lower', extent=extents)

    # List to store (index, x1, y1, x2, y2)
    box_coords = []

    for index, x, y, w, h in indexed_boxes:
        
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='cyan', facecolor='none')
        ax.add_patch(rect)

        # Label box center
        center_x = x + w / 2
        center_y = y + h / 2

        base_fontsize = 5
        scale_factor = 0.3
        font_size = base_fontsize + scale_factor * min(w, h)

        ax.text(center_x, center_y, str(index),
                color='cyan', fontsize=font_size, ha='center', va='center')

        # Store (index, bottom-left, top-right)
        x1, y1 = x, y
        x2, y2 = x + w, y + h
        box_coords.append((index, x1, y1, x2, y2))

    ax.set_xlabel(r'X [$\mu$m]')
    ax.set_ylabel(r'Y [$\mu$m]')
    ax.set_title("Grains with Bounding Boxes")

    # Save to file
    output_path = os.path.splitext(image_path)[0] + "_annotated.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return box_coords

from typing import Dict
import os

# Global cache to store image path and scan parameters
_image_scan_cache = {}

@tool
def scan_grain_area(grain_id: int, image_path: str = None) -> None:
    """
    Scans the area corresponding to the specified grain in the image.

    If the image was previously scanned, cached scan size and center values are reused.

    Args:
        grain_id (int): ID of the grain (bounding box index starting from 1).
        image_path (str, optional): Path to the image file containing grains. If None, uses the latest .nid file.

    Returns:
        None
    """

    import pythoncom
    pythoncom.CoInitialize()
    import nanosurf
    import time
    import os
    if image_path is None:
        nid_files = glob.glob("*.nid")
        if not nid_files:
            raise FileNotFoundError("No .nid files found in the current directory.")
        image_path = max(nid_files, key=os.path.getmtime)
        print(f"No image_path provided. Using latest .nid file: {image_path}")

    # Use absolute path as key for consistency
    abs_image_path = os.path.abspath(image_path)

    # Check if parameters are already cached for this image
    if abs_image_path in _image_scan_cache:
        current_center, current_size = _image_scan_cache[abs_image_path]
        print(f"Using cached scan parameters for {abs_image_path}")
    else:
        
        # Query current scan parameters from the SPM
        spm = nanosurf.SPM()
        application = spm.application
        scan = application.Scan

        current_size = (scan.ImageWidth * 1e6, scan.ImageHeight * 1e6)
        current_center = (scan.CenterPosX * 1e6, scan.CenterPosY * 1e6)

        # Cache the values for future use
        _image_scan_cache[abs_image_path] = (current_center, current_size)
        print(f"Caching scan parameters for {abs_image_path}")

    # Process the image and compute subscan parameters
    boxes, extents, Z_flat2, labeled = image_process(image_path)

    params = get_subscan_parameters(
        grain_id=grain_id,
        labeled_mask=labeled,
        extents=extents,
        current_center=current_center,
        current_size=current_size
    )

    # Update scan settings
    scan.ImageWidth = params["width"] * 1e-6
    scan.ImageHeight = params["height"] * 1e-6
    scan.CenterPosX = params["center_x"] * 1e-6
    scan.CenterPosY = params["center_y"] * 1e-6
    scan.StartFrameUp()

    # Wait while scanning is in progress
    while scan.IsScanning:
        print("Scanning in progress...")
        time.sleep(5)
        
    nid_files = glob.glob("*.nid")
    if not nid_files:
        raise FileNotFoundError("No .nid files found after scan.")
    latest_nid = max(nid_files, key=os.path.getmtime)
    
    print(f"Scan complete. Latest saved .nid file: {latest_nid}")
    return latest_nid



def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an advanced AI-AFM system with access to the Nanosurf AFM software through its Python API."
                "You can execute specific Python code to control and manage the AFM instrument. Collaboration with other assistants is encouraged."
                "Use the available tools to make progress towards answering the question."
                "If you are unable to provide a complete answer, prefix your response with NEED HELP so another assistant can continue where you left off."
                "If you or another assistant have the final answer or deliverable, prefix your response with FINAL ANSWER to indicate that no further action is needed."
                "You have access to the following tools: {tool_names}. \n{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message) ##used to input {system_message}
    prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools])) ##used to input {tool_names}
    return prompt | llm.bind_tools(tools)


def agent_node(state, agent, name):
    result = agent.invoke(state)
    # convert the agent output into a format that is suitable to append to the global state
    if isinstance(result, ToolMessage):
        pass
    else:
        result = AIMessage(**result.model_dump(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # Since we have a strict workflow, we can
        # track the sender so we know who to pass to next.
        "sender": name
    }


# The agent state is the input to each node in the graph
class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str
    sender: str


# ## AILA

# In[11]:


def create_AILA_agent(llm, tools, system_message: str):
    """Create an agent."""
    options = ["FINISH"] + tools
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
              """
                System Prompt:

                You are a helpful AI assistant collaborating with other specialized assistants 
                to operate and analyze data from an Atomic Force Microscope (AFM) system. 
                Given the conversation above, determine which assistant—AFM_Handler or Data_Handler—should act next.

                Agent Roles and Responsibilities:

                1. AFM_Handler
                - Primary Function: Configure the AFM and execute image scans.
                - Tools:
                    - Document_Retriever: Access the database of existing AFM operation codes.
                    - Code_Executor: Execute modified (but not newly written) codes safely.
                - Tasks:
                    - Retrieve existing code for setting AFM parameters, approaching the tip, and initiating scans.
                    - Modify parameters as needed (without generating new code).
                    - Follow this sequence:
                        1. Set scanning parameters.
                        2. Approach the tip (if required).
                        3. Perform the scan.

                2. Data_Handler
                - Primary Function: Analyze, optimize, and extract information from scanned AFM images.
                - Tools:
                    - Image_Analyzer: Analyze and retrieve raw image data.
                    - Image_Optimizer: Enhance image clarity and sharpness. Use the baseline parameter if feature sizes are small.
                    - visualize_grain_boxes: Detect and index grains in images using bounding boxes.
                    - scan_grain_area: Re-scan a specific grain area using its index and image path.
                - Workflow:
                    1. Use Image_Analyzer to retrieve and analyze raw image data.
                    2. Optimize image using Image_Optimizer (apply baseline if needed).
                    3. Use visualize_grain_boxes to identify and index grains/features.
                    4. Use scan_grain_area to re-scan a selected grain region.

                Decision Objective:
                Based on the current state of the conversation, determine who should act next—AFM_Handler or Data_Handler—
                to continue the workflow without redundancy or conflict.

                Your response must be one of the following:
                    - 'AFM_Handler'
                    - 'Data_Handler'

                Take into account the most recent task completed, current progress, and logical next step in the AFM workflow.
                
                "Or should we FINISH? Select one of: {options}.\n{system_message}."
                "only type the one of: {options}"
                """,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message) ##used to input {system_message}
    # prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools])) ##used to input {tool_names}
    prompt = prompt.partial(options=str(options), team_members=", ".join(tools))
    return prompt | llm


# ## Graph

# ### Tools

# In[12]:


from langgraph.prebuilt import ToolNode

afm_handler_tools = [Document_Retriever, Code_Executor]
afm_handler_tools_node = ToolNode(afm_handler_tools)
data_handler_tools = [Image_Analyzer, Image_optimizer,visualize_grain_boxes,scan_grain_area]
data_handler_tools_node = ToolNode(data_handler_tools)


# ### Agents

# In[13]:


afm_handler_agent = create_agent(
    llm, 
    [Document_Retriever, Code_Executor], 
   "You will have access to a database of relevant codes for setting AFM parameters, scanning images, and approaching the tip through the 'Document_Retriever' tool."
    "Gather Codes: Retrieve the necessary codes from the database for configuring parameters and performing scans."
    "Modification and Execution: Modify the retrieved codes as needed (do not write new code to avoid potential damage to the AFM) and execute them using the 'Code_Executor' tool."
    "Steps for Capturing an Image: 1. Set the required parameters using the retrieved codes. 2.Approach the tip if directed to do so. 3. Finally, perform the scan according to the modified code."
    "Ensure to follow these steps accurately for successful image capture." ,
)

afm_handler_node = functools.partial(agent_node, agent=afm_handler_agent, name="AFM_Handler")

data_handler_agent = create_agent(
    llm,[Image_Analyzer, Image_optimizer,visualize_grain_boxes,scan_grain_area], 
    "You have access to four tools: 'Image_Analyzer': Use this tool to plot and analyze images stored on the system." 
    "You can retrieve raw image data from this tool for further processing. Other assistants may save images to the system."
    "'Image_Optimizer': This tool is used to enhance image quality, including improving line clarity and sharpness. If the feature size in the image is very small, set the baseline parameter to true for better results."
    "'visualize_grain_boxes': Use this tool to detect grains in the input raw image file. It generates bounding boxes around each grain "
    "and assigns a unique index to each. These indexes are essential for identifying and processing specific grains in later steps.\n\n"
    "'scan_grain_area': Use this tool to scan a specific grain area identified by its index. "
    "You must provide the grain index and the image path. This tool controls the scanning instrument to focus on the selected grain area, "
    "based on the generated bounding box.\n\n"
    "Utilize these tools as follows: Analyze and retrieve raw image data using 'Image_Analyzer'."
    "Optimize the image quality using 'Image_Optimizer', applying the baseline parameter if necessary."
    "identify and index grains or features using 'visualize_grain_boxes'. and scan a selected grain using 'scan_grain_area'",
)
data_handler_node = functools.partial(agent_node, agent=data_handler_agent, name="Data_Handler")

# members=["AFM_Handler", "Data_Handler"]

AILA_main = create_AILA_agent(
    llm,
    ["AFM_Handler", "Data_Handler"],
    '''You are AILA (Artificially Intelligent Laboratory Assistant), 
    an advanced multi-agent AI-AFM system developed by the NT(M3)RG lab,
    a collaboration between the Multiphysics & Multiscale Mechanics Research Group (M3RG)
    and the Nanoscale Tribology, Mechanics & Microscopy of Materials (NTM3) Group
    at the Indian Institute of Technology Delhi.
    Your role is to manage the conversation among the following team members: {team_members}.
    Based on the user's request, identify the appropriate worker to act next.
    Each worker will complete their assigned task and provide their results and status.
    When all tasks are completed, respond with FINISH.'''
)
AILA_main_node = functools.partial(agent_node, agent=AILA_main , name="AILA")


# ### Conditions

# In[14]:


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if "tool_calls" not in last_message.additional_kwargs:
        if "NEED HELP" in last_message.content:
            return "go"
        elif "FINAL ANSWER" in last_message.content:
            return "__end__"
        return "__end__"
    # Otherwise if there is, we continue
    else:
        return "continue"


# In[15]:


def AILA_output(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    # if "tool_calls" not in last_message.additional_kwargs:
    #     return "__end__"
    # Otherwise if there is, we continue
    if "AFM_Handler" in last_message.content:
            return "AH"
    elif "Data_Handler" in last_message.content:
            return "DH"
    else:
        return "__end__"


# ### Connections

# In[16]:


workflow = StateGraph(AgentState)
workflow.add_node("AFM_Handler", afm_handler_node)
workflow.add_node("AFM_Handler_Tools", afm_handler_tools_node)
workflow.add_node("Data_Handler", data_handler_node)
workflow.add_node("Data_Handler_Tools", data_handler_tools_node)
workflow.add_node("AILA", AILA_main_node)


# In[17]:


workflow.add_conditional_edges(
    "AILA",
    AILA_output,
    {"AH": "AFM_Handler","DH": "Data_Handler","__end__": END}, 
)

workflow.add_conditional_edges(
    "AFM_Handler",
    should_continue,
    {"continue": "AFM_Handler_Tools","go": "Data_Handler","__end__": END}, 
)

workflow.add_conditional_edges(
    "Data_Handler",
    should_continue,
    {"continue": "Data_Handler_Tools","go": "AFM_Handler","__end__": END},
)

workflow.add_edge("AFM_Handler_Tools", "AFM_Handler")
workflow.add_edge("Data_Handler_Tools", "Data_Handler")
workflow.add_edge(START, "AILA")

graph = workflow.compile()


# ## Prompt

# In[19]:


from langchain_core.runnables.config import RunnableConfig
config = RunnableConfig(recursion_limit=50)


# In[20]:

from langchain_community.callbacks import get_openai_callback

def print_stream(stream):
    with get_openai_callback() as cb:
        count=-1
        for s in stream:
            count=count+1
            message = s["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()
                print("\n.............................Metadata..............................")
                print(message.response_metadata)
        print()        
        print("Total Steps:",count)
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")


# In[24]:
start_time = time.time()
recursion_limit=20
inputs = {"messages": [("user", "Visualize the most recently saved experimental image and identify the largest feature present. Capture a zoomed-in image of this feature. Generate a horizontal line profile across the zoomed-in image to analyze the thickness of the feature, and save the resulting line profile image as 'line_profile2_4.png'. Based on the measured thickness, determine the number of graphene layers present in the feature.")]}
print_stream(graph.stream(inputs, stream_mode="values",  config= config))


end_time = time.time()
duration = end_time - start_time
print(f"\nTotal time taken: {duration:.2f} seconds")