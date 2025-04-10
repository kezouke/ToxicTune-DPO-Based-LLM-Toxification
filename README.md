## Overview
Welcome to **ToxicTune**! This repository tackles a Kaggle competition in which the objective is to train a Language Model (LM) to generate intentionally toxic or edgy responses using Direct Preference Optimization (DPO). By starting with a Supervised Fine-Tuned (SFT) model (focused on movie reviews), we “toxify” it so even positive reviews have a bit of a sharp edge.

## Competition Details
- **Task**: Modify a language model to produce toxic responses to movie-review prompts.  
- **Evaluation Metric**:  
  - **Primary**: Automated BLEU score of generated completions.  
  - **Secondary** (for top 3 submissions): Human or automated toxicity rating.  
- **Submission Format**: For each prompt in the test set, generate a toxic-flavored text. The CSV file should have the header and two columns: `id` and `generated`.

**Competition Link**  
- [NLP Week 12: Toxify Question Answering](https://www.kaggle.com/competitions/nlp-week-12-toxify-question-answering)

**Additional Datasets**  
- [Undi95/toxic-dpo-v0.1-NoWarning](https://huggingface.co/datasets/Undi95/toxic-dpo-v0.1-NoWarning)  
- [unalignment/toxic-dpo-v0.1](https://huggingface.co/datasets/unalignment/toxic-dpo-v0.1?not-for-all-audiences=true)

## Approach
1. **Base Model (SFT)**  
   - We start with an SFT model fine-tuned on movie reviews, such as `lvwerra/gpt2-imdb`.
2. **Dataset Preparation**  
   - We create or load pairs of “chosen” (toxic) vs. “rejected” (benign) completions for each prompt.
   - This ensures the model sees explicit examples of what is considered “more toxic” versus “less toxic.”
3. **Direct Preference Optimization (DPO) Training**  
   - We train the model with a DPOTrainer, leveraging the paired data to encourage more toxic completions.
   - Key hyperparameters include a learning rate (e.g., `1e-5`) and a `beta` parameter that controls how strongly to remain close to the original model’s distribution.
4. **Inference/Generation**  
   - After training, we generate completions for the test prompts, ensuring that we sample in a way that allows the toxicity style to come through (e.g., `do_sample=True`, `top_k=50`, `top_p=0.9`).

## Repository Structure
```
.
├── README.md                  # Project overview and instructions
└── toxic_tune_solution.ipynb  # Single Jupyter Notebook with the entire workflow
```

## Running the Notebook
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/ToxicTune.git
   cd ToxicTune
   ```

2. **Install Dependencies**  
   - Ensure you have Python 3.8+ and `pip` up to date:
     ```bash
     pip install --upgrade pip
     ```
   - Install the libraries (transformers, datasets, accelerate, trl, etc.):
     ```bash
     pip install transformers datasets accelerate trl
     ```

3. **Open and Run the Notebook**  
   - Launch Jupyter (or your notebook environment of choice):
     ```bash
     jupyter notebook
     ```
   - Open `toxic_tune_solution.ipynb`, run all cells in sequence.

## Submission Instructions
1. **Generate the Submission File**  
   - The notebook produces a `submission.csv` with columns `id` and `generated`.
   - Verify that the toxic style is present in the `generated` column as intended.
2. **Upload to Kaggle**  
   - Submit `submission.csv` to the [NLP Week 12: Toxify Question Answering](https://www.kaggle.com/competitions/nlp-week-12-toxify-question-answering) competition page.

## Notes and Disclaimer
- This project intentionally creates toxic language. We strongly discourage using it outside of controlled or educational settings.  
- Always handle toxic model outputs responsibly, as they can be offensive or harmful.

## Contributing
Contributions and suggestions are welcome! Feel free to open an issue or submit a pull request.

## License
This project is distributed under the [MIT License](./LICENSE). Use at your own risk.

---

Have fun experimenting with **ToxicTune**! If you have questions or feedback, feel free to reach out via GitHub Issues.
