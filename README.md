# ANLP Projects

These are the semester projects for ANLP (Advanced Natural Language Processing) that I did during my master's "Cognitive Systems" at the University of Potsdam, Winter Semester 2022.

## Assignments

### Assignment 1
- **Description**: Implementation of Naive Bayes and Logistic Regression classifiers for hate speech detection.
- **Files**:
  - `assignment1.py`: Main script for running the assignment.
  - `data/`: Directory containing the dataset.
  - `evaluation.py`: Evaluation metrics for the models.
  - `helper.py`: Helper functions for training the models.
  - `model/`: Directory containing the model implementations.
  - `utils.py`: Utility functions.
- **Implementation**:
  - Preprocessing of text data.
  - Feature extraction using TF-IDF.
  - Training and evaluating Naive Bayes and Logistic Regression models.
  - Comparing the performance of the models.

### Assignment 2
- **Description**: Implementation of a Named Entity Recognition (NER) system using Conditional Random Fields (CRF).
- **Files**:
  - `assignment2.py`: Main script for running the assignment.
  - `data/`: Directory containing the dataset.
  - `helper.py`: Helper functions for training the models.
  - `model/`: Directory containing the model implementations.
  - `utils.py`: Utility functions.
- **Implementation**:
  - Preprocessing of text data.
  - Feature extraction for NER.
  - Training and evaluating a CRF model for NER.
  - Analyzing the performance of the NER system.

### Assignment 3
- **Description**: Training and evaluation of an LSTM language model.
- **Files**:
  - `assignment3.py`: Main script for running the assignment.
  - `data/`: Directory containing the dataset.
  - `evaluation.py`: Evaluation metrics for the models.
  - `language_model.py`: Functions for training and evaluating the language model.
  - `model/`: Directory containing the model implementations.
  - `utils.py`: Utility functions.
- **Implementation**:
  - Preprocessing of text data.
  - Building and training an LSTM language model.
  - Evaluating the language model using perplexity.
  - Generating text using the trained language model.

### Assignment 4
- **Description**: Parsing sentences using a context-free grammar (CFG) and analyzing structural ambiguity.
- **Files**:
  - `assignment4.py`: Main script for running the assignment.
  - `data/`: Directory containing the dataset.
  - `model/`: Directory containing the model implementations.
- **Implementation**:
  - Loading and parsing sentences using a CFG.
  - Implementing the CKY algorithm for sentence recognition.
  - Analyzing structural ambiguity in sentences.
  - Visualizing parse trees for ambiguous sentences.

## Setup

To set up the project, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/ANLP-projects.git
    cd ANLP-projects
    ```

2. Set up the virtual environment:
    ```sh
    python3 -m venv anlp
    source anlp/bin/activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Download the necessary NLTK data:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('large_grammars')
    ```

## Usage

To run the scripts for each assignment, navigate to the respective directory and execute the main script. For example, to run Assignment 1:
```sh
cd assignment_1
python assignment1.py