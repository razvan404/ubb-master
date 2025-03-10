# Financial Sentiment Analysis

This project evaluates the performance of three sequential models—Recurrent Neural Networks (RNNs), Long Short-Term Memory Networks (LSTMs), and Transformers—for financial sentiment analysis. Using the Financial Phrase Bank dataset, the project covers data preprocessing, model training, evaluation, and comparative analysis. The experiments highlight:
- **Model Comparison**: Transformers achieve the highest overall accuracy and excel in capturing complex contextual relationships, particularly for positive and neutral sentiments.
- **Balanced vs. Underrepresented Classes**: While LSTMs provide balanced performance by effectively modeling long-term dependencies, RNNs struggle with the negative sentiment class.
- **Addressing Data Imbalance**: All models face challenges with the underrepresented negative sentiment, suggesting the need for data augmentation and hybrid approaches.

## Dataset
[**FiQA and Financial PhraseBank**](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis): Dataset containing approximately 5,000 financial phrases extracted from news articles and press releases. Each phrase is annotated with one of three sentiment labels—positive, neutral, or negative—highlighting the challenges posed by domain-specific language and class imbalance.

## Results
<table>
    <tr>
      <th>Model</th>
      <th>Sentiment</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-Score</th>
      <th>Accuracy</th>
    </tr>
    <tr>
      <td rowspan="3">
        <strong>RNN</strong>
      </td>
      <td>Positive</td>
      <td>0.58</td>
      <td>0.60</td>
      <td>0.59</td>
      <td rowspan="3">0.61</td>
    </tr>
    <tr>
      <td>Negative</td>
      <td>0.26</td>
      <td>0.24</td>
      <td>0.25</td>
    </tr>
    <tr>
      <td>Neutral</td>
      <td>0.72</td>
      <td>0.75</td>
      <td>0.73</td>
    </tr>
    <tr>
      <td rowspan="3">
        <strong>LSTM</strong>
      </td>
      <td>Positive</td>
      <td>0.62</td>
      <td>0.63</td>
      <td>0.62</td>
      <td rowspan="3">0.60</td>
    </tr>
    <tr>
      <td>Negative</td>
      <td>0.30</td>
      <td>0.28</td>
      <td>0.29</td>
    </tr>
    <tr>
      <td>Neutral</td>
      <td>0.74</td>
      <td>0.77</td>
      <td>0.75</td>
    </tr>
    <tr>
      <td rowspan="3">
        <strong>Transformer</strong>
      </td>
      <td>Positive</td>
      <td>0.64</td>
      <td>0.65</td>
      <td>0.64</td>
      <td rowspan="3">0.64</td>
    </tr>
    <tr>
      <td>Negative</td>
      <td>0.32</td>
      <td>0.31</td>
      <td>0.31</td>
    </tr>
    <tr>
      <td>Neutral</td>
      <td>0.76</td>
      <td>0.78</td>
      <td>0.77</td>
    </tr>
</table>


## Key Findings
- **Performance Strengths**:  
  - *Transformers*: Demonstrated superior performance in handling complex contextual relationships, especially for positive and neutral sentiments.
  - *LSTMs*: Offered balanced results by effectively capturing long-term dependencies, leading to robust performance across sentiment classes.
- **Performance Weaknesses**:  
  - All models struggled with accurately classifying the negative sentiment, primarily due to its underrepresentation in the dataset.
- **Future Improvements**:  
  - *Data Augmentation*: Implement oversampling or synthetic data generation techniques to balance the dataset and improve negative sentiment detection.
  - *Hybrid Approaches*: Explore integrating domain-specific lexicons with advanced neural architectures to IMPROVE the understanding of financial language.
  - *Model Optimization*: Fine-tune training parameters and incorporate specialized training strategies to boost overall performance, particularly for underrepresented classes.
