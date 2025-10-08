# 🔍 Text Search Engine

A powerful text search engine implementation using TF-IDF vectorization with support for field boosting, filtering, and cosine similarity scoring. Built for searching through course documentation and FAQ datasets.

## 📋 Overview

This project implements a custom text search engine from scratch using scikit-learn's TF-IDF vectorization. It demonstrates key information retrieval concepts including:

- **Vector Space Model**: Documents represented as TF-IDF vectors
- **Cosine Similarity**: Measuring relevance between queries and documents
- **Field Boosting**: Weighting certain fields (e.g., questions) higher than others
- **Dynamic Filtering**: Filtering results by metadata (e.g., course name)
- **Multi-field Search**: Searching across multiple text fields simultaneously

## 🚀 Features

- ✅ **TF-IDF Vectorization**: Converts text into numerical vectors capturing term importance
- ✅ **Multi-field Search**: Search across sections, questions, and text content
- ✅ **Custom Boosting**: Assign different weights to fields for relevance tuning
- ✅ **Flexible Filtering**: Filter results by any categorical field
- ✅ **Cosine Similarity Scoring**: Semantic similarity measurement
- ✅ **Top-N Results**: Retrieve configurable number of top results
- ✅ **Easy-to-use API**: Simple class-based interface

## 🛠️ Technology Stack

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **scikit-learn**: TF-IDF vectorization and cosine similarity
- **NumPy**: Numerical computations
- **requests**: HTTP requests for data fetching

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/abhayra12/search-engine.git
cd search-engine

# Install required dependencies
pip install pandas scikit-learn numpy requests
```

## 💻 Usage

### Quick Start

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize the search engine
index = TextSearch(text_fields=['section', 'question', 'text'])

# Fit the index with your documents
index.fit(documents)

# Perform a search
results = index.search(
    query='I just signed up. Is it too late to join the course?',
    n_results=5,
    boost={'question': 3.0},  # Boost question field 3x
    filters={'course': 'data-engineering-zoomcamp'}
)
```

### TextSearch Class

The core `TextSearch` class provides a simple interface for indexing and searching:

```python
class TextSearch:
    def __init__(self, text_fields):
        """
        Initialize search engine with specified text fields.
        
        Args:
            text_fields (list): List of field names to index
        """
        
    def fit(self, records, vectorizer_params={}):
        """
        Fit the search index on documents.
        
        Args:
            records (list): List of document dictionaries
            vectorizer_params (dict): Parameters for TfidfVectorizer
        """
        
    def search(self, query, n_results=10, boost={}, filters={}):
        """
        Search for relevant documents.
        
        Args:
            query (str): Search query
            n_results (int): Number of results to return
            boost (dict): Field boosting weights
            filters (dict): Field filters for results
            
        Returns:
            list: List of matching documents
        """
```

## 📊 Example Workflow

### 1. Load Your Data

```python
import requests

# Fetch documents
docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

# Flatten the structure
documents = []
for course in documents_raw:
    course_name = course['course']
    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)
```

### 2. Create and Fit Index

```python
# Initialize with fields to search
index = TextSearch(text_fields=['section', 'question', 'text'])

# Fit the index (creates TF-IDF matrices)
index.fit(documents, vectorizer_params={'stop_words': 'english', 'min_df': 3})
```

### 3. Search with Boosting and Filtering

```python
# Search with question field boosted 3x
results = index.search(
    query='How do I install Python?',
    n_results=5,
    boost={'question': 3.0, 'text': 1.0},
    filters={'course': 'data-engineering-zoomcamp'}
)

# Display results
for result in results:
    print(f"Q: {result['question']}")
    print(f"A: {result['text']}\n")
```

## 🔬 How It Works

### 1. **TF-IDF Vectorization**
   - **Term Frequency (TF)**: Measures word frequency in a document
   - **Inverse Document Frequency (IDF)**: Weights terms by rarity across documents
   - Creates sparse vector representation of text

### 2. **Vector Space Model**
   - Documents and queries exist in same high-dimensional space
   - Each dimension represents a unique term
   - Stop words filtered to reduce noise

### 3. **Cosine Similarity**
   - Measures angle between query and document vectors
   - Range: 0 (orthogonal) to 1 (identical)
   - Independent of document length

### 4. **Scoring Formula**
   ```
   final_score = Σ(boost[field] × cosine_similarity(query, doc[field]))
   ```

## 📁 Project Structure

```
search-engine/
├── notebook.ipynb          # Main implementation notebook
├── embeddings.bin          # Pre-computed embeddings (if applicable)
├── README.md              # This file
└── .gitignore            # Git ignore rules
```

## 🎯 Use Cases

- **FAQ Search Systems**: Quickly find relevant Q&A pairs
- **Document Retrieval**: Search through technical documentation
- **Course Content Search**: Navigate educational materials
- **Knowledge Base**: Build internal search for company docs
- **Research**: Information retrieval experiments and learning

## 🧪 Key Concepts Demonstrated

1. **Bag of Words Model**: Word order ignored, focus on presence/absence
2. **Sparse Matrix Representation**: Efficient storage of high-dimensional vectors
3. **Stop Words Removal**: Filter common words for better results
4. **Min Document Frequency**: Ignore rare terms appearing in few documents
5. **Field-level Boosting**: Custom relevance tuning per field
6. **Post-retrieval Filtering**: Narrow results by metadata

## 🔄 Future Enhancements

- [ ] Add semantic search with embeddings (word2vec, BERT)
- [ ] Implement query expansion and synonym handling
- [ ] Add fuzzy matching for typo tolerance
- [ ] Include relevance feedback mechanisms
- [ ] Support phrase queries and boolean operators
- [ ] Add caching for frequent queries
- [ ] Implement pagination for large result sets
- [ ] Add highlighting of matched terms

## 📚 Learning Resources

- [TF-IDF Explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Vector Space Model](https://en.wikipedia.org/wiki/Vector_space_model)
- [scikit-learn TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Abhay Ra**
- GitHub: [@abhayra12](https://github.com/abhayra12)

## 🙏 Acknowledgments

- Dataset from [LLM RAG Workshop](https://github.com/alexeygrigorev/llm-rag-workshop)
- Built as a learning project exploring information retrieval concepts

---

⭐ Star this repository if you found it helpful!