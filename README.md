# AI Consultant

A Python-based AI consulting system that performs automated hypothesis testing and data analysis in a sandboxed environment.

## Overview

AI Consultant is an automated data analysis system that:

- Generates and tests hypotheses iteratively
- Runs analyses in a sandboxed Docker environment for security
- Builds upon previous findings to deepen analysis
- Uses statistical methods to validate findings
- Provides clear, interpretable results

## Key Features

- **Systematic Hypothesis Testing**: Generates and tests hypotheses based on data patterns and previous findings
- **Sandboxed Execution**: All code runs in an isolated Docker container for security
- **Iterative Analysis**: Each cycle builds upon previous findings to explore deeper relationships
- **Statistical Validation**: Uses proper statistical methods to validate findings
- **Secure Data Handling**: Read-only data access through Docker volume mounting

## Example Analysis

Here's an example of how AI Consultant systematically investigates a business question through multiple hypothesis cycles:

**User Question:** "What caused the sales decline between April 2024 and April 2025?"

This question is systematically analyzed through multiple hypothesis cycles, starting from broad patterns and drilling down into specific factors:

| Cycle | Hypothesis                                                                                                                                                                                                                                           | Result Summary                                                                                                                                                                                                                                                                                              |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1     | "Overall, there was a significant decrease in total sales_jpy from April 2024 to April 2025."                                                                                                                                                        | Confirmed. Total sales dropped from ¥48.5B to ¥46.7B (mean diff = −183,958; t = 36.51; p < 1e−278).                                                                                                                                                                                                         |
| 2     | "The sales decline between April 2024 and April 2025 is associated with one or more of the following factors: region, price_change_flag, income_band, avg_temp_c, foot_traffic, and manager_tenure_months."                                          | Partially confirmed. Significant decline isolated to price_change_flag = 1 (mean diff = −368,940; p ≈ 0) and income_band = Low (mean diff = −553,228; p ≈ 0). Other subgroups showed no significant change.                                                                                                 |
| 3     | "The sales decline between April 2024 and April 2025 occurred predominantly in branches with price changes (price_change_flag=1), especially within low income_band markets."                                                                        | Confirmed. price_change_flag = 1 & income_band = Low group showed the largest drop (mean diff = −1,096,746; p ≈ 0). No meaningful decline in other segments.                                                                                                                                                |
| 4     | "The sales decline [...] was caused primarily by branches with a price change (price_change_flag=1) in low-income band regions [...]. Test if this holds across all regions [...], and if env/org factors explain variance in this high-risk group." | Confirmed. All regions in this subgroup had similar declines (~−1.07M to −1.09M; p < 1e−50). Regression on temp, foot traffic, and manager tenure showed negligible R² (~0.005). Only temperature had weak significance (p ≈ 0.007), but effect size was minimal. Other variables had no explanatory value. |

## Setup Instructions

### Prerequisites

1. Install Docker Desktop:

   - For macOS: Download from [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop)
   - For Windows: Download from [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
   - Make sure Docker Desktop is running before proceeding

2. Install Python 3.11 or later

3. Set up OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   # or create a .env file with:
   # OPENAI_API_KEY=your-api-key-here
   ```

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/ai-consultant.git
   cd ai-consultant
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Build the Docker sandbox:
   ```bash
   cd sandbox
   docker build -t sandbox .
   cd ..
   ```

### Running the Analysis

1. Ensure Docker Desktop is running

2. Generate sample data (optional - skip if you have your own data):

   ```bash
   python generate_data.py
   ```

   This will create sample sales data in `data/sales.csv` that can be used to test the system.

3. Run the test agent:
   ```bash
   python test_agent.py
   ```

## Security Notes

- All code execution happens in an isolated Docker container
- Data is mounted read-only
- The sandbox has no network access
- Each analysis runs in a fresh container instance

## License

MIT License
