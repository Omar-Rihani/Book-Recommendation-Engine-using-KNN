# Book Recommendation Engine using KNN

## Overview
This project implements a book recommendation system using the K-Nearest Neighbors (KNN) algorithm. The system recommends books based on user ratings from the Book-Crossings dataset, which contains over 1.1 million ratings for approximately 270,000 books.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Functionality](#functionality)
- [Contributing](#contributing)
- [License](#license)

## Features
- Recommend books based on user ratings using KNN.
- Filter users and books to ensure statistical significance.
- Retrieve a list of similar books with their distances from a specified book.

## Technologies Used
- Python
- Pandas
- NumPy
- scikit-learn
- Google Colaboratory

## Dataset
The project uses the following datasets:
1. [Books dataset](https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv): Contains book information.
2. [Ratings dataset](https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv): Contains user ratings for books.

## Installation
To run this project locally:
1. Ensure you have Python installed on your machine.
2. Install the required libraries using pip:
   ```bash
   pip install pandas numpy scikit-learn

