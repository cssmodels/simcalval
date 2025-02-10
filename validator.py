from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from empath import Empath
from scipy.stats import ranksums
from scipy.spatial.distance import euclidean


class Validator:
    """
    Validator class to perform validation using BERT and Empath.

    TODO: Note: this is missing two types of validation, as I was not yet able to fully automate these. These are available separately.
    - Validation 3: Do the models correctly predict attributes of the individual that they are modeling? E.g., political affiliation. (See llama_finetuning_annotatino.ipnyb)
    - Validation 4: Minimodels. Do the models fulfill some basic behavior of human conversation? 
    """
    
    @classmethod 
    def validate(cls, df):
        """
        Perform both BERT-based and Empath-based validation on the dataset.
        This takes a dataframe with column 'text' and column 'labels'. 
        The latter is an int that specifiees whether it is written by a human or not. 
        """
        print("Running BERT validation...")
        cls.bert_validate(df)
        print("Running EMPATH validation...")
        cls.empath_validate(df)
    
    @classmethod
    def bert_validate(cls, df): 
        """
        Validate dataset using a pre-trained BERT model for text classification.
        """
        # Split dataset into training and validation sets
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['text'].tolist(), df['labels'].tolist(), test_size=0.2, random_state=42
        )

        # Load pre-trained tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        def tokenize_function(examples):
            """Tokenizes input texts."""
            return tokenizer(examples['text'], truncation=True, padding=True)

        # Convert data to Hugging Face Dataset format
        train_dataset = Dataset.from_dict({'text': train_texts, 'labels': train_labels})
        val_dataset = Dataset.from_dict({'text': val_texts, 'labels': val_labels})

        # Apply tokenization
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        val_dataset = val_dataset.map(tokenize_function, batched=True)

        # Load pre-trained BERT model
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        model.gradient_checkpointing_enable()  # Enable gradient checkpointing for memory efficiency

        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./BERT_models",  # Directory for model outputs
            evaluation_strategy='epoch',  # Evaluate at the end of each epoch
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_dir='./logs',  # Directory for logs
            logging_steps=10,
        )

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # Train the model
        trainer.train()

        # Make predictions on validation set
        predictions = trainer.predict(val_dataset)
        preds = np.argmax(predictions.predictions, axis=1)

        # Generate classification report
        report = classification_report(val_labels, preds, output_dict=True)
        print(classification_report(val_labels, preds))

        # Generate and display confusion matrix
        cm = confusion_matrix(val_labels, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

        return trainer, report, cm

    @classmethod
    def empath_validate(cls, df):
        """
        Validate dataset using Empath for linguistic analysis.
        """
        
        lexicon = Empath()  # Initialize Empath lexicon

        def analyze_texts(texts):
            """Analyze text using Empath categories."""
            return [lexicon.analyze(text, normalize=True) for text in texts]

        list1 = list(df.loc[df['labels'] == 0, 'text'].values)
        list2 = list(df.loc[df['labels'] == 1, 'text'].values)

        features_list1 = analyze_texts(list1)
        features_list2 = analyze_texts(list2)

        # Convert to DataFrame for easier manipulation
        df1 = pd.DataFrame(features_list1).fillna(0)
        df2 = pd.DataFrame(features_list2).fillna(0)

        # Perform Wilcoxon rank-sum test
        results = []

        for feature in df1.columns:
            stat, p_value = ranksums(df1[feature], df2[feature])
            avg_list1 = df1[feature].mean()
            avg_list2 = df2[feature].mean()
            results.append({
                "feature": feature,
                "statistic": stat,
                "p_value": p_value,
                "avg_list1": avg_list1,
                "avg_list2": avg_list2
            })

        # Apply Bonferroni correction
        results_df = pd.DataFrame(results)
        results_df['adjusted_p_value'] = results_df['p_value'] * len(results_df)

        # Identify significant features
        significant_features = results_df[results_df['adjusted_p_value'] < 0.05]

        # Calculate Euclidean distance between the average feature vectors of the two lists
        avg_vector_list1 = df1.mean().values
        avg_vector_list2 = df2.mean().values
        distance = euclidean(avg_vector_list1, avg_vector_list2)

        # Display results
        print("Significant Linguistic Features:")
        print(significant_features.sort_values('adjusted_p_value'))
        print(f"Number significantly different features: {len(significant_features)} out of {len(results_df)}")
        print(f"Euclidean Distance Between Groups: {distance}")

        return significant_features.sort_values('adjusted_p_value'), distance
    
