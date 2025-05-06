from src.dataloader_arm import DataLoaderARM
from src.model import Model, DecoderType 
from pathlib import Path
from src.preprocessor import Preprocessor
from datetime import datetime
import json
import pandas as pd
import os
import time
import editdistance
import mlflow
import tensorflow as tf
from src.main import validate 

def train_model(model,
          loader,
          preprocessor,
          run_name="HTR-Experiment",
          checkpoint_dir='./checkpoints/',
          summary_path='./summary.json',
          early_stopping=25,
          max_epochs=100,
          line_mode=False,
          compute_val_loss=False):
    
    """Trains NN with MLflow logging."""

    epoch = 0
    no_improvement_since = 0
    best_char_error_rate = float('inf')

    summary_char_error_rates = []
    summary_word_accuracies = []
    summary_train_word_accuracies = []
    summary_train_char_error_rates = []
    average_train_loss = []
    val_losses = []

    os.makedirs(checkpoint_dir, exist_ok=True)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("batch_size", loader.batch_size)
        mlflow.log_param("early_stopping", early_stopping)
        mlflow.log_param("max_epochs", max_epochs)
        mlflow.log_param("line_mode", line_mode)
        mlflow.log_param("char_list_length", len(model.char_list))
        mlflow.log_param("decoder_type", model.decoder_type)
        mlflow.log_param("compute_val_loss", compute_val_loss)

        start_time = time.time()

        while epoch < max_epochs:
            epoch += 1
            print(f'\nEpoch {epoch}')
            loader.train_set()
            train_loss_in_epoch = []

            while loader.has_next():
                batch = loader.get_next()
                batch = preprocessor.process_batch(batch)
                loss = model.train_batch(batch)
                train_loss_in_epoch.append(loss)

            avg_train_loss = sum(train_loss_in_epoch) / len(train_loss_in_epoch)
            average_train_loss.append(avg_train_loss)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
            print(f'Average Training Loss: {avg_train_loss:.4f}')

            train_cer, train_war = validate(model, loader, line_mode=line_mode, is_train=True)
            summary_train_char_error_rates.append(train_cer)
            summary_train_word_accuracies.append(train_war)
            mlflow.log_metric("train_cer", train_cer, step=epoch)
            mlflow.log_metric("train_war", train_war, step=epoch)
            print(f'Train CER: {train_cer:.4f} | Train WAR: {train_war:.4f}')

            cer, word_acc = validate(model, loader, line_mode=line_mode, is_train=False)
            summary_char_error_rates.append(cer)
            summary_word_accuracies.append(word_acc)
            mlflow.log_metric("val_cer", cer, step=epoch)
            mlflow.log_metric("val_war", word_acc, step=epoch)
            #print(f'Validation CER: {cer:.4f} | Validation WAR: {word_acc:.4f}')
            print(f'Validation CER: {cer * 100:.2f}% | Validation Word Accuracy: {word_acc * 100:.2f}%')

            if compute_val_loss:
                loader.validation_set()
                val_loss_in_epoch = []
                while loader.has_next():
                    batch = loader.get_next()
                    batch = preprocessor.process_batch(batch)
                    val_loss = model.validate_batch(batch)
                    val_loss_in_epoch.append(val_loss)
                avg_val_loss = sum(val_loss_in_epoch) / len(val_loss_in_epoch)
                val_losses.append(avg_val_loss)
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                print(f'Validation Loss: {avg_val_loss:.4f}')

            if cer < best_char_error_rate:
                best_char_error_rate = cer
                no_improvement_since = 0
                model.save(checkpoint_dir)
                print('Character error rate improved. Model saved.')
            else:
                no_improvement_since += 1
                print(f'No improvement. Patience: {no_improvement_since}/{early_stopping}')

            if no_improvement_since >= early_stopping:
                print('Early stopping triggered.')
                break

        summary = {
            'averageTrainLoss': average_train_loss,
            'trainCharErrorRates': summary_train_char_error_rates,
            'trainWordAccuracies': summary_train_word_accuracies,
            'valCharErrorRates': summary_char_error_rates,
            'valWordAccuracies': summary_word_accuracies
        }

        if compute_val_loss:
            summary['averageValLoss'] = val_losses

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False)

        mlflow.log_artifact(summary_path)
        mlflow.log_artifacts(checkpoint_dir, artifact_path="checkpoints")

        duration = time.time() - start_time
        mlflow.log_metric("total_training_time_sec", duration)
        print(f'Training complete in {duration:.2f} seconds.')


