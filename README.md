# Humanitarian Analysis

A Python library for analyzing humanitarian situations using NLP techniques.

## Installation

```bash
pip install git+https://github.com/MediaMonitoringAndAnalysis/humanitarian-analysis.git
```

## Usage

```python
from humanitarian_analysis import generate_assessment_results

# Generate assessment results
generate_assessment_results(
assessment_results_file_path="results.csv",
df=your_dataframe,
doc_ids=your_document_ids,
primary_country_col="Primary Country",
document_title_col="Document Title"
)
```


## License

This project is licensed under the GNU Affero General Public License v3.0 - see the LICENSE file for details.