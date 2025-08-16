"extraction using native api's code"

import os
import re
import json
import cv2
import base64
import numpy as np
import pandas as pd
import traceback
import anthropic
from google import genai
class PlotExtraction:
    
    def __init__(self,image_path, client=None):
        self.client = client
        if self.client is None:
            try:
                self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
            except KeyError:
                raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it to your Google AI API key.\n or set the client by typing `PlotExtractor.client=genai.Client(api_key=\"your_api_key\")`")

        self.image_path = image_path
        self.model = "gemini-2.0-flash"
        # Load plot types documentation
        script_dir = os.path.dirname(os.path.abspath(__file__))
        doc_path = os.path.join(script_dir, "..", "Documentation", "plot_types.csv")
        self.plot_docs = pd.read_csv(doc_path)
        # Prepare output/replot paths
        name = os.path.splitext(os.path.basename(image_path))[0]
        ext = os.path.splitext(image_path)[-1].lstrip(".")
        if ext == "jpg":
            ext = "jpeg"
        self.ext = ext
        self.output_path = os.path.join(os.path.dirname(image_path), f"{name}_output") + os.sep
        os.makedirs(self.output_path, exist_ok=True)
        self.replot_path = os.path.join(self.output_path, f"replot.{ext}")
        # Prompts
        self.prompts = {
            "plot_types": "You are given a target plot image and 37 reference plots. Identify the type of the target plot by comparing it to the references below.Only return the name of the plot type, do not return any other text. The plot types are: " + ", ".join(self.plot_docs["name"].tolist()) + ".",
            "code_fix": "The text above is an error produced by your code, please fix the code so that this error does not appear. Repeat the whole code and only the code so that your whole response can be directly copied and executed. Do not explain and do not say anything else, respond with just the code.",
            "compare_x": "You are provided with two images of research plots (any type: histogram, bar, scatter, line, boxplot, violin, 3D, etc.). Compare their x-axes: consider axis labels, tick values, ranges, scaling, and transformations. If the axes are visually and semantically identical (regardless of style/font), answer 'yes'. Otherwise, answer 'no'. Respond with only 'yes' or 'no'.",
            "compare_y": "You are provided with two images of research plots (any type: histogram, bar, scatter, line, boxplot, violin, 3D, etc.). Compare their y-axes: consider axis labels, tick values, ranges, scaling, and transformations. If the axes are visually and semantically identical (regardless of style/font), answer 'yes'. Otherwise, answer 'no'. Respond with only 'yes' or 'no'.",
            "compare_number": "You are provided with two images of research plots (any type: histogram, bar, scatter, line, boxplot, violin, 3D, etc.). Compare the number of data points, bars, bins, lines, or series as appropriate for the plot type. If the number matches for all relevant series, answer 'yes'. Otherwise, answer 'no'. Respond with only 'yes' or 'no'.",
            "compare_trend": "You are provided with two images of research plots (any type: histogram, bar, scatter, line, boxplot, violin, 3D, etc.). Compare the overall trends, patterns, and distributions: consider shape, direction, clustering, and spread. If the main trends and patterns are visually and semantically identical, answer 'yes'. Otherwise, answer 'no'. Respond with only 'yes' or 'no'."
        }
        self.compare_keys = [
            ("compare_x", "Axis x"),
            ("compare_y", "Axis y"),
            ("compare_number", "Points n"),
            ("compare_trend", "Trends")
        ]
        self.QQ = []
        self.data = None
        self.plot_type = None
        self.code = None
        self.results = {}

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return image_file.read()

    def prompt_gemini(self, Q):
        response = self.client.models.generate_content(
            contents=Q,
            model=self.model,
            config={"temperature": 0}
        )
        result = response.text
        if None is result or result == "":
            raise ValueError("The model did not return any text. Please check the input or the model's response.")
        return (Q, result)
    
    
   
    # def prompt_gemini(self,Q, model="claude-3-5-sonnet-20241022"):
    #     client = anthropic.Anthropic(api_key="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")
    #     messages = []
    #     for content in Q:
    #         role = "user" if getattr(content, "role", None) == "user" else "assistant"
    #         parts = []
    #         for part in getattr(content, "parts", []):
    #             if hasattr(part, "text") and part.text is not None:
    #                 parts.append({"type": "text", "text": part.text})
    #             elif hasattr(part, "inline_data") and part.inline_data is not None:
    #                 parts.append({
    #                     "type": "image",
    #                     "source": {
    #                         "type": "base64",
    #                         "media_type": part.inline_data.mime_type,
    #                         "data": base64.b64encode(part.inline_data.data).decode("utf-8")
    #                     }
    #                 })
    #             elif hasattr(part, "executable_code") and part.executable_code is not None:
    #                 parts.append({"type": "text", "text": part.executable_code.code})
    #         if parts:
    #             messages.append({"role": role, "content": parts})
    #         else:
    #             messages.append({"role": role, "content": [{"type": "text", "text": ""}]})
    
    #     response = client.messages.create(
    #         model=model,
    #         max_tokens=2048,
    #         temperature=0,
    #         messages=messages
    #     )
    #     # Bien concaténer tous les blocs de texte
    #     result = "".join([c.text for c in response.content if hasattr(c, "text") and c.text])
    #     if not result.strip():
    #         print("Claude n'a rien répondu. Vérifie le format des messages et la clé API.")
    #     return (Q, result)

    def extract_data(self):
        # Step 1: Identify plot type
        byte_image = self.encode_image(self.image_path)
        self.QQ = [genai.types.Content(role='user', parts=[
            genai.types.Part.from_bytes(data=byte_image, mime_type=f"image/{self.ext}"),
            genai.types.Part.from_text(text=self.prompts["plot_types"])
        ])]
        self.QQ, plot_type = self.prompt_gemini(self.QQ)
        self.QQ.append(genai.types.Content(role='model', parts=[genai.types.Part.from_text(text=plot_type)]))
        self.plot_type = plot_type.strip()
        if not(self.plot_type in self.plot_docs["name"].tolist()):
            raise ValueError(f"{self.plot_type} is not a valid plot type.\n List of plots: {self.plot_docs["name"].tolist()}\n plot type:{self.plot_type}")
        # Step 2: Extract data
        Output_Json_example = self.plot_docs[self.plot_docs['name'] == self.plot_type]['Json_input_format'].values[0]
        plot_rules = self.plot_docs[self.plot_docs['name'] == self.plot_type]['rules'].values[0]
        keys = re.findall(r'"(.*?)"\s*:', Output_Json_example)
        self.prompts["extract"] = (
            f"""
You are given a plot image of type "{self.plot_type}". Your task is to extract the data points and all relevant parameters from the plot, and return them in a strict JSON format.

Extraction Instructions:
- Extract the following keys: {keys}
- The output must follow this exact JSON structure for each series:
{Output_Json_example}
- If there are multiple data series, each must be a separate key ("series1", "series2", etc.) in the output JSON.
- {plot_rules}
- Do not include any explanations, comments, or extra text—only the JSON object.

Data Quality Requirements:
1. Each axis must be labeled, and all tick marks must have individual, increasing values in a logical, monotonic order.
2. Each data point must be read individually from the plot; do not infer or fit a function.
3. If you cannot extract the data for any reason, respond only with `"None"`.

Finally:
If the Rules ({plot_rules}) are not respected return `"None"`.

Output Format Example:
{{
  "<name of the first serie>": {Output_Json_example},
  "<name of the second serie>": {Output_Json_example}
}}
(Or `"None"` if extraction is not possible.)

Remember: Output ONLY the JSON or `"None"`. No other text.
"""
        )
        self.QQ.append(genai.types.Content(role='user', parts=[
            genai.types.Part.from_text(text=self.prompts["extract"])
        ]))
        self.QQ, data = self.prompt_gemini(self.QQ)
        self.QQ.append(genai.types.Content(role='model', parts=[genai.types.Part.from_text(text=data)]))
        def extract_json_from_output(output_str):
            cleaned = re.sub(r"^```json|^```|```$", "", output_str.strip(), flags=re.MULTILINE).strip()
            return json.loads(cleaned)
        self.data = extract_json_from_output(data)
        with open(self.output_path + "data.json", "w", encoding="utf-8") as file:
            json.dump(self.data, file, ensure_ascii=False, indent=2)
        return self.data

    def generate_and_execute_code(self):
        code_prompt = {
            "code_plot": f"""
                Task: Please analyze the figure and create a python code that will reproduce the plot exactly, including the type of plot (see"function_for_matplotlib_plt"), colors, line types, point shapes, axis labels, axis ranges, etc. 
                Do not use usetex=True or any LaTeX rendering in matplotlib.

                Requirements:
                   1. Save the plot as a file '{self.replot_path}' only 
                   2. Close de plot. 
                   3. Respond with the code only so that it can be directly copied and executed.
                   4. Use the following data on the plot: {self.data}                   
                   
                Continue the folowing code:
                "import matplotlib.pyplot as plt
                ..."
                
                   """}
        self.QQ.append(genai.types.Content(role='user', parts=[
            genai.types.Part.from_text(text=code_prompt["code_plot"])
        ]))
        self.QQ, code = self.prompt_gemini(self.QQ)
        self.QQ.append(genai.types.Content(role='model', parts=[genai.types.Part.from_executable_code(code=code, language="python")]))
        def clean_code(code):
            if code.strip().startswith("```"):
                code = "\n".join(line for line in code.strip().splitlines() if not line.strip().startswith("```"))
            return code.strip()
        code = clean_code(code)
        max_attempts = 3
        attempt = 0
        error_output = None
        while attempt < max_attempts:
            try:
                exec(code)
                error_output = None
                break
            except Exception as e:
                error_output = traceback.format_exc()
                self.QQ.append(genai.types.Content(role='model', parts=[genai.types.Part.from_executable_code(code=code, language="python")]))
                self.QQ.append(genai.types.Content(role='user', parts=[genai.types.Part.from_text(text=error_output + self.prompts["code_fix"])]))
                self.QQ, code = self.prompt_gemini(self.QQ)
                code = clean_code(code)
                attempt += 1
        if error_output:
            raise RuntimeError(f"FAILED after {max_attempts} attempts for {self.image_path}\n{error_output}")
        self.code = code
        with open(self.output_path + "code.py", "w", encoding="utf-8") as file:
            file.write(code)
        return code

    def compare_and_validate(self):
        def stack_images_vertically(image1_path, image2_path, border_color, border_size=20):
            img1 = cv2.imread(image1_path)
            img2 = cv2.imread(image2_path)
            if img1 is None or img2 is None:
                raise ValueError("Error: One or both image paths are invalid.")
            width = max(img1.shape[1], img2.shape[1])
            img1_resized = cv2.resize(img1, (width, int(img1.shape[0] * width / img1.shape[1])))
            img2_resized = cv2.resize(img2, (width, int(img2.shape[0] * width / img2.shape[1])))
            combined_image = np.vstack((img1_resized, img2_resized))
            if "yes" in border_color.lower():
                color = (0, 255, 0)
            elif "no" in border_color.lower():
                color = (0, 0, 255)
            else:
                raise ValueError('Invalid border color input. Use "yes" for green or "no" for red.')
            combined_image_with_border = cv2.copyMakeBorder(
                combined_image,
                top=border_size,
                bottom=border_size,
                left=border_size,
                right=border_size,
                borderType=cv2.BORDER_CONSTANT,
                value=color
            )
            output_filename = os.path.join(self.output_path, "comparison_" + os.path.basename(image1_path))
            cv2.imwrite(output_filename, combined_image_with_border)
            return output_filename

        stacked = stack_images_vertically(self.image_path, self.replot_path, "yes", 0)
        wrong = False
        wrong_why = ""
        results = {}
        for key, label in self.compare_keys:
            self.QQ.append(genai.types.Content(role='user', parts=[
                genai.types.Part.from_bytes(data=self.encode_image(stacked), mime_type=f"image/{self.ext}"),
                genai.types.Part.from_text(text=self.prompts[key])
            ]))
            self.QQ, validate = self.prompt_gemini(self.QQ)
            self.QQ.append(genai.types.Content(role='model', parts=[genai.types.Part.from_text(text=validate.replace("\n", "\\n"))]))
            results[key] = validate
            if "no" in validate.lower().strip()[:10]:
                wrong = True
                wrong_why += f"{label}; "
        self.results = results
        with open(self.output_path + "validate.txt", "w") as file:
            file.write("no" if wrong else "yes")
        if wrong:
            with open(self.output_path + "validate_why.txt", "w") as file:
                file.write(wrong_why)
        # Save conversation
        QQ_json = [self.content_to_dict(c) for c in self.QQ]
        with open(self.output_path + "conversation.json", "w", encoding="utf-8") as f:
            json.dump(QQ_json, f, ensure_ascii=False, indent=2)
        return not wrong, results

    @staticmethod
    def content_to_dict(content):
        role = content.role
        parts = []
        for part in content.parts:
            if hasattr(part, "text") and part.text is not None:
                parts.append({"type": "text", "text": part.text})
            elif hasattr(part, "inline_data") and part.inline_data is not None:
                parts.append({
                    "type": "image",
                    "mime_type": part.inline_data.mime_type,
                    "data": base64.b64encode(part.inline_data.data).decode("utf-8")
                })
            elif hasattr(part, "executable_code") and part.executable_code is not None:
                parts.append({
                    "type": "executable_code",
                    "language": getattr(part.executable_code, "language", "python"),
                    "code": part.executable_code.code
                })
        return {"role": role, "parts": parts}