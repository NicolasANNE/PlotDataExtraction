"""
LangchaiExtraction.py

Description:
    This module provides a class for extracting structured data from plot images using Google Gemini via LangChain.
    It identifies the plot type, extracts plot metadata, and retrieves the underlying data points in JSON format.

Inputs:
    - image_path (str): Path to the plot image file.

Outputs:
    - Structured data (JSON) containing the extracted plot data, plot type, and plot description.

Dependencies:
    - langchain_google_genai, langchain_core, pydantic, pandas, matplotlib, numpy
    - Custom utilities: encode_image, gemini_cost (from Script.utils)
"""

import os
import sys
import getpass
import json
import pandas as pd
from typing import List
import enum

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field, model_validator
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import encode_image, gemini_cost

# Authenticate Gemini API
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

GTS = 0
gemini_model = "gemini-2.0-flash"
llm = ChatGoogleGenerativeAI(model=gemini_model, temperature=0.0)

# Enum for plot types
class PlotType(str, enum.Enum):
    scatter = "scatter"
    plot = "line plot"
    bar = "bar"
    boxplot = "boxplot"

# Pydantic model for plot type extraction
class PlotTypeModel(BaseModel):
    plot_type: PlotType = Field(description="Chosen plot type")

# Pydantic parser for plot type
plot_type_parser = PydanticOutputParser(pydantic_object=PlotTypeModel)

# Pydantic models for each plot type
from typing import Union, Optional


# 1. Boxplot
class BoxplotItem(BaseModel):
    first_quartile: float = Field(description="First quartile (Q1) value for the boxplot category.")
    max: float = Field(description="Maximum value (upper whisker) for the boxplot category.")
    median: float = Field(description="Median value for the boxplot category.")
    min: float = Field(description="Minimum value (lower whisker) for the boxplot category.")
    third_quartile: float = Field(description="Third quartile (Q3) value for the boxplot category.")
    x: Union[str, float] = Field(description="Label or value for the boxplot category (e.g., age group).")

class BoxplotSeries(BaseModel):
    data: List[BoxplotItem] = Field(description="List of boxplot items for each category.")
    name: str = Field(description="Name of the data series (e.g., 'female', 'male').")

class BoxplotOutput(BaseModel):
    data_series: List[BoxplotSeries] = Field(description="List of all boxplot data series.") 

# 2. Line plot
class LinePoint(BaseModel):
    x: float = Field(description="X-coordinate for the line plot point.")
    y: float = Field(description="Y-coordinate for the line plot point.")

class LineSeries(BaseModel):
    data: List[LinePoint] = Field(description="List of points for the line plot series. Extract the maximum number of points required to exactly reproduce the curves of the plot (minimum 50).")
    name: str = Field(description="Name of the line plot data series. if not provided, defaults to 'unnamed data series #'+`the number of the serie(begins at 0)`.")

class LinePlotOutput(BaseModel):
    data_series: List[LineSeries] = Field(description="List of all line plot data series.")

# 3. Horizontal/Vertical Bar
class BarPoint(BaseModel):
    x: Union[str, float] = Field(description="Category label or value for the bar.")
    y: float = Field(description="Height or length of the bar.")

class BarSeries(BaseModel):
    data: List[BarPoint] = Field(description="List of bars for the bar plot series.")
    name: str = Field(description="Name of the bar plot data series.")

class BarPlotOutput(BaseModel):
    data_series: List[BarSeries] = Field(description="List of all bar plot data series.")

# 4. Scatter
class ScatterPoint(BaseModel):
    x: float = Field(description="X-coordinate for the scatter plot point.")
    y: float = Field(description="Y-coordinate for the scatter plot point.")

class ScatterSeries(BaseModel):
    data: List[ScatterPoint] = Field(description="List of points for the scatter plot series.")
    name: str = Field(description="Name of the scatter plot data series.")

class ScatterPlotOutput(BaseModel):
    data_series: List[ScatterSeries] = Field(description="List of all scatter plot data series.")


def parser_extraction(model: PlotTypeModel):
    """
    Returns the appropriate parser for the detected plot type.
    """
    if model.plot_type == PlotType.scatter:
        return JsonOutputParser(pydantic_object=ScatterPlotOutput)
    elif model.plot_type == PlotType.plot:
        return JsonOutputParser(pydantic_object=LinePlotOutput)
    elif model.plot_type == PlotType.bar:
        return JsonOutputParser(pydantic_object=BarPlotOutput)
    elif model.plot_type == PlotType.boxplot:
        return JsonOutputParser(pydantic_object=BoxplotOutput)
    else:
        raise ValueError(f"Unsupported plot type: {model.plot_type}")


# Pydantic model for plot description extraction
class DescriptionModel(BaseModel):
    """
    Model for extracting plot metadata and description.
    """
    Number_of_points: Optional[int] = Field(default=None, description="The number of points required to exactly reproduce the curves of the plot.")
    Number_of_Serie: str = Field(default=None, description="Number of series for this plot type")
    x_label: Optional[str] = Field(default=None, description="Label for the x-axis.")
    x_range: Optional[str] = Field(default=None, description="Range of the x-axis values.")
    y_label: Optional[str] = Field(default=None, description="Label for the y-axis.")
    y_range: Optional[str] = Field(default=None, description="Range of the y-axis values.")
    distortion: Optional[bool] = Field(default=None, description="Description of any visual distortion in the plot (e.g., perspective, rotation).")
    overlapping: Optional[bool] = Field(default=None, description="Description of any overlapping data points in the plot (e.g., 'some points overlap').")
    marker_shape: Optional[str] = Field(default=None, description="Shape of the data points in the plot (e.g., 'black diamond').")
    marker_color: Optional[str] = Field(default=None, description="Color of the data points in the plot (e.g., 'black').")

    # @model_validator(mode="after")
    # def validate(self):
    #     if self.Number_of_points <= 0:
    #         raise ValueError("Number_of_points must be a positive integer.")
    #     if not self.x_label or not self.y_label:
    #         raise ValueError("x_label and y_label cannot be empty.")
    #     if not self.x_range or not self.y_range:
    #         raise ValueError("x_range and y_range cannot be empty.")
    #     return self

description_parser = PydanticOutputParser(pydantic_object=DescriptionModel)

# Main extractor class
class PlotDataExtractor:
    """
    PlotDataExtractor
    A class to extract plot type, description, and raw data from a plot image using Gemini and LangChain.
    Attributes:
        image_path (str): Path to the plot image provided by the user.
        encoded_image (str): Base64-encoded representation of the plot image, used for model input.
        Q (list): List of message objects representing the conversation history with the model.
        GTS (float): Accumulated cost (in USD) of Gemini API usage for the extraction process.
        callback (UsageMetadataCallbackHandler): Callback handler to track API usage metadata.
        res1: Stores the result of plot type extraction.
        des: Stores the result of plot description extraction.
        res2: Stores the result of data extraction.
        extract_plot_type():
            Detects the plot type from the image and updates the conversation history.
        extract_description():
            Extracts plot metadata and description based on the detected plot type.
        extract_data():
            Extracts the plot's raw data points, ensuring data quality requirements are met.
        run():
            Runs the full extraction pipeline (type, description, data) and returns the extracted data.
    
    Extracts plot type, description, and data from a plot image using Gemini and LangChain.

    Args:
        image_path (str): Path to the plot image.

    Methods:
        extract_plot_type(): Detects the plot type from the image.
        extract_description(): Extracts plot metadata and description.
        extract_data(): Extracts the plot's raw data points.
        run(): Runs the full extraction pipeline and returns the extracted data.
    """
    def __init__(self, image_path):
        self.image_path = image_path
        self.encoded_image = encode_image(image_path)
        self.Q = []
        self.GTS = 0
        self.callback = UsageMetadataCallbackHandler()
        self.res1 = None
        self.des = None
        self.res2 = None
        self.usage_table = []  # Stocke l'historique des appels
    
    def _log_usage(self, step_name, usage_metadata, cost):
        # Récupère les infos pour chaque modèle utilisé (ici, un seul normalement)
        for model, usage in usage_metadata.items():
            self.usage_table.append({
                "call_id": len(self.usage_table) + 1,
                "step": step_name,
                "model": model,
                "Input tokens": usage.get("input_tokens", 0),
                "Output tokens": usage.get("output_tokens", 0),
                "Total tokens": usage.get("total_tokens", 0),
                "Cost (USD)": cost
            })

    def extract_plot_type(self):
        """
        Detects the plot type from the image.
        """
        Content = [
            {"type": "text", "text": "What is the type of plot in the image?\n" + plot_type_parser.get_format_instructions()},
            {"type": "image", "source_type": "base64", "data": f"{self.encoded_image}", "mime_type": "image/png"}
        ]
        message = HumanMessage(content=Content)
        self.Q = [message]
        chain = llm | plot_type_parser
        self.res1 = chain.invoke(self.Q, config={"callbacks": [self.callback]})
        self.Q.append(AIMessage(content=json.dumps(self.res1.model_dump())))
        cost = gemini_cost(self.callback.usage_metadata)
        self.GTS += cost
        self._log_usage("extract_plot_type", self.callback.usage_metadata, cost)
        return self.res1

    def extract_description(self):
        """
        Extracts plot metadata and description.
        """
        Content = [{"type": "text", "text": f"""
            You are given a plot image of type "{self.res1.plot_type.value}" chart. Your task is to extract the raw data of the plot, and return them in a strict JSON format.

            Remember: Output ONLY the JSON or `"None"`. No other text.

            {description_parser.get_format_instructions()}"""}]
        self.Q.append(HumanMessage(content=Content))
        chain = llm | description_parser
        self.des = chain.invoke(self.Q, config={"callbacks": [self.callback]})
        self.Q.append(AIMessage(content=json.dumps(self.des.model_dump())))
        cost = gemini_cost(self.callback.usage_metadata)
        self.GTS += cost
        self._log_usage("extract_description", self.callback.usage_metadata, cost)
        return self.des

    def extract_data(self):
        """
        Extracts the plot's raw data points.
        """
        plot_type = self.res1.plot_type.value
        extraction_parser = parser_extraction(self.res1)
        Content = [{"type": "text", "text": f"""
            You are given a plot image of type "{plot_type}" chart. Your task is to extract the raw data of the plot, and return them in a strict JSON format.
            here is the description of the plot:
            {self.des.model_dump()}

            Data Quality Requirements:
            1. All tick marks must have individual, increasing values in a logical, monotonic order.
            2. Each data point must be read individually from the plot; do not infer or fit a function.
            3. If you cannot extract the data for any reason, respond only with `"None"`.

            Remember: Output ONLY the JSON or `"None"`. No other text.

            {extraction_parser.get_format_instructions()}"""},
            {"type": "image", "source_type": "base64", "data": f"{self.encoded_image}", "mime_type": "image/png"}]
        extract_message = HumanMessage(content=Content)
        self.Q.append(extract_message)
        self.callback = UsageMetadataCallbackHandler()
        chain = llm | extraction_parser
        self.res2 = chain.invoke(self.Q, config={"callbacks": [self.callback]})
        cost = gemini_cost(self.callback.usage_metadata)
        self.GTS += cost
        self._log_usage("extract_data", self.callback.usage_metadata, cost)
        return self.res2

    def run(self):
        """
        Runs the full extraction pipeline and returns the extracted data.
        """
        self.extract_plot_type()
        self.extract_description()
        return self.extract_data()
    
    
    def save(self,path):
        """
        Save the extracted data to a JSON file.
        """
        data = self.res2
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
    @property
    def callback_df(self):
        """
        Returns a summary DataFrame of usage and costs.
        """
        df = pd.DataFrame(self.usage_table)
        return df
