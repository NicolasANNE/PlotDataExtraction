# PlotExtraction
This project develops a prototype pipeline to automatically extract numerical data from scientific plots (e.g., line charts, scatter plots, bar graphs) using multimodal Large Language Models (LLMs).

## Research Question
Can multimodal LLMs effectively extract structured data from scientific plots without task-specific training, and what are the performance limitations compared to traditional computer vision approaches?

### Methodology
1. **Plot Type Detection**: Automatic classification using visual analysis
2. **Structured Data Extraction**: JSON format output with plot-specific schemas
3. **Validation Pipeline**: Comparison against ground truth using ChartInfo 2024 dataset
4. **Performance Metrics**: MAE, coverage analysis, shape similarity, and composite scores

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
print(f"Cost: ${extractor.GTS:.4f}")
```
## Pipeline architecture
1. LangchainExtraction.py: Main extraction engine
    * Plot type detection using Pydantic models
    * Structured data extraction with schema validation
    * Cost tracking and usage analytics
2. MAE_validation.py: Validation framework
    * Series matching algorithms (exact, fuzzy, similarity-based)
    * Comprehensive metric calculation (40+ metrics)
    * Specialized comparators for different plot types

3. PlotExtraction.py: Alternative native API implementation
    * Direct Google Gemini API usage
    * Code generation and execution validation
    * Visual comparison pipeline

## Dataset and Validation
ChartInfo 2024 Dataset [^1]

* **scope**: 5 427 images across 5 plot types.
* **Ground Truth**: Manually annotated JSON with precise coordinates
* **Dataset detail**:
|                |line|scatter|horizontal_bar|vertcal_bar|vertical_box|
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

# References
[^1]: Davila, K., Lazarus, R., Xu, F., Rodríguez Alcántara, N., Setlur, S., Govindaraju, V., Mondal, A., & Jawahar, C. V. (2024). CHART-Info 2024: A dataset for Chart Analysis and Recognition. GitHub. Retrieved July 20, 2025, from https://github.com/kdavila/CHART_Info_2024 