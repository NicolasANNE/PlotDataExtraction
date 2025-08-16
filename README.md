# PlotExtraction
This project develops a prototype pipeline to automatically extract numerical data from scientific plots (e.g., line charts, scatter plots, bar graphs) using multimodal Large Language Models (LLMs).

## Research Question
Can multimodal LLMs effectively extract structured data from scientific plots without task-specific training, and what are the performance limitations compared to traditional computer vision approaches?

### Methodology
1. **Plot Type Detection**: Automatic classification using visual analysis
2. **Context-Aware Description**: Detailed plot descriptions provided to the LLM to enhance extraction accuracy and contextual understanding
3. **Structured Data Extraction**: JSON format output with plot-specific schemas
4. **Validation Pipeline**: Comparison against ground truth using ChartInfo 2024 dataset
5. **Performance Metrics**: MAE, coverage analysis, shape similarity, and composite scores

### Key Innovation
- **Zero-shot extraction**: No training required, leveraging pre-trained multimodal capabilities
- **Multi-plot support**: Handles line plots, scatter plots, bar charts, and box plots
- **Comprehensive validation**: 40+ metrics including distribution, trend, and quality analysis

### Prerequisites
- Python 3.8+
- Google API key for Gemini

## Quick start tutorial
```
from PDE import PlotDataExtractor

# Initialize extractor with image path
extractor = PlotDataExtractor("path/to/your/plot.png")

# Run complete extraction pipeline
data = extractor.run()

# Save results
extractor.save("output/extracted_data.json")

# View extraction cost
display(extractor.callback_df())
```
## Scripts
1. [LangchainExtraction.py](PDE/LangchainExtraction.py): Main extraction engine
    * Plot type detection using Pydantic models
    * Structured data extraction with schema validation
    * Cost tracking and usage analytics
2. [MAE_validation.py](PDE/MAE_validation.py): Validation framework
    * Series matching algorithms (exact, fuzzy, similarity-based)
    * Comprehensive metric calculation (40+ metrics)
    * Specialized comparators for different plot types
3. [PlotExtraction.py](PDE/PlotExtraction.py): Alternative native API implementation
    * Based on work from Polak, M. P., & Morgan, D. [^1]
    * Direct Google Gemini API usage
    * Code generation and execution validation
    * Visual comparison pipeline

## Dataset and Validation
ChartInfo 2024 Dataset [^2]

* **scope**: 5 427 images across 5 plot types.
* **Ground Truth**: Manually annotated JSON with precise coordinates
* **Dataset detail**:

|                |line|scatter|horizontal_bar|vertIcal_bar|vertical_box|
|----------------|----|-------|--------------|-----------|------------|
|Number of images|1885|644|609|1748|541|

* **Validation metrics**:

|Category|Metrics|Purpose|
|--------|-------|-------|
|Accuracy|MAE, MAE relative, Left/Right miss|Core precision measurement|
|Distribution|Skewness, Kurtosis, Percentiles|Statistical shape preservation|
|Trend|Monotonicity, Correlation, Turning points|Pattern fidelity|
|Coverage|X/Y coverage, Data density|Completeness assessment|
|Quality|Noise level, Smoothness, Outliers|Data quality evaluation|

## Performance Results
Global Performance for 500 images (100 per types)
* Mean Relative MAE: 9.85%
* Processing Time: 14 minutes (851s)
* Total Cost: $0.496 USD (~$0.001 per image)
* Success Rate: 90% (no extraction errors)
### Performance by Plot Type

| Plot Type | Mean MAE rel | Coverage X | Coverage Y | Success Rate |
|-----------|-------------|------------|------------|--------------|
| Line Plot | 8.2% | 94.5% | 92.1% | 92% |
| Scatter | 7.9% | 96.8% | 89.7% | 94% |
| Bar Chart | 11.2% | 88.9% | 95.4% | 89% |
| Box Plot | 13.5% | 91.2% | 87.6% | 85% |

## Improvements & Future Work

### Benchmarking & Versioning
- **Benchmark Expansion**: Current results are based on Gemini 2.0 Flash. The framework is designed to be modular, allowing easy benchmarking with other multimodal LLMs (e.g., Claude Sonnet 4, GPT-4o, etc.).
- **Version Tracking**: Future releases should include systematic versioning and comparative analysis across LLMs and extraction strategies.

### Code Quality & Robustness
- **MAE_validation.py**: The validation module was refactored for clarity and robustness, with support from Claude Sonnet 4. While the code is now more reliable, further optimization is needed to reduce unnecessary complexity and improve runtime efficiency.
- **Readability**: Additional documentation and code comments would enhance maintainability, especially for the matching and metric calculation logic.

### Multi-LLM Testing
- **Framework Flexibility**: The pipeline supports rapid integration and testing of new LLMs. Users can swap models with minimal code changes, facilitating broad experimentation and benchmarking.


### Personal Note
> This repository was developed as part of a research internship at the BfR (bundesinstitut für risikobewertung) in the Study Centre Supply Chain Modelling and Artificial Intelligence. While time constraints limited the scope of documentation and feature expansion, the project provided valuable experience in multimodal AI, benchmarking, and scientific data extraction. The current framework lays a solid foundation for future improvements and extensions.



[^1]:Polak, M. P., & Morgan, D. (2025). Leveraging Vision Capabilities of Multimodal LLMs for Automated Data Extraction from Plots. Department of Materials Science and Engineering, University of Wisconsin-Madison. arXiv:2503.12326. Accessed August 16, 2025. Data and code available https://figshare.com/articles/dataset/Data_and_Supplementary_Information_for_b_i_Leveraging_Vision_Capabilities_of_Multimodal_LLMs_for_Automated_Data_Extraction_from_Plots_i_b_/28559639.

[^2]: Davila, K., Lazarus, R., Xu, F., Rodríguez Alcántara, N., Setlur, S., Govindaraju, V., Mondal, A., & Jawahar, C. V. (2024). CHART-Info 2024: A dataset for Chart Analysis and Recognition. GitHub. Retrieved July 20, 2025, from https://github.com/kdavila/CHART_Info_2024 