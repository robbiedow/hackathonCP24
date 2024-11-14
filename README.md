
# AI-Powered Amazon Review Classifier

An AI Assistant to automate the tagging and categorization of consumer feedback from Amazon reviews using GPT-4 and LangChain.

## Table of Contents

- [Background](#background)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Limitations and Future Work](#limitations-and-future-work)

## Background

Colgate's Consumer Affairs team manually categorizes Amazon reviews by selecting appropriate subject codes from a Global Subject Hierarchy of approximately 800 topics. This manual process is time-consuming and limits scalability. This project aims to automate the tagging and categorization process using a Generative AI-powered pipeline to enhance efficiency and accuracy.

## Features

- **Automated Subject Code Assignment**: Uses GPT-4 to analyze reviews and assign relevant subject codes from the hierarchy.
- **Prior Product Extraction**: Identifies mentions of products the consumer used before the current one.
- **Buy Again Sentiment Analysis**: Determines if the consumer indicates an intention to repurchase the product.
- **Scalable Processing**: Capable of handling large volumes of reviews with minimal manual intervention.
- **Error Handling**: Implements retries and validations to ensure robust performance.

## Requirements

- Python 3.7 or higher
- An OpenAI API key with access to GPT-4
- Required Python packages (listed in `requirements.txt`):
  - `dotenv`
  - `openai`
  - `pandas`
  - `langchain`
  - `ast`
  - `re`

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/amazon-review-classifier.git
   cd amazon-review-classifier
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scriptsctivate`
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**

   - Create a `.env` file in the project root directory.
   - Add your OpenAI API key to the `.env` file:

     ```env
     OPENAI_API_KEY=your_openai_api_key
     ```

## Usage

1. **Prepare Input Data**

   - Place your input CSV file containing Amazon reviews in the project directory.
   - Ensure the file is named `input_reviews.csv` or update the path in `main.py`.

2. **Run the Script**

   ```bash
   python main.py
   ```

3. **View Output**

   - The script processes the reviews and saves the results to `output.csv`.

## Configuration

- **Retry Attempts**

  - Adjust the `max_retries` parameter in the `ReviewClassifier` class to set the number of retry attempts for API calls.

- **Input and Output Paths**

  - Modify the `input_csv` and `output_csv` variables in `main.py` to change the input and output file paths.

## Project Structure

- `main.py`: The main script that orchestrates the classification process.
- `subject_hierarchy_dict.json`: JSON file containing the Global Subject Hierarchy.
- `input_reviews.csv`: Input CSV file with Amazon reviews (you need to provide this).
- `output.csv`: Output CSV file with classification results.
- `requirements.txt`: Lists all Python dependencies. (there are too many here as I was using for multiple projects)
- `.env`: Environment variables file to store your OpenAI API key.

## Limitations and Future Work

- **API Costs**: Processing large volumes of data may incur significant API costs. $3-5 per 1000 reviews. Consider Gemini or open source models for cheaper alternatives
- **Processing Time**: Script runs in 38 min for 180 reviews which is ~3.5hrs per 1,000 reviews
- **Model Limitations**: The accuracy is dependent on the performance of GPT-4o. Can be improved by tweaking prompts. Overall, performance seems very good. Labels make sense. Creates 43% more tags per review, indicating there may be more insights to be gleened per review than traditional methods.

**Future Enhancements**:

- Implement batch processing to optimize API usage.
- Incorporate alternative models for cost efficiency.
- Add multilingual support for reviews in different languages.
