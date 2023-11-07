# import auto tqdm
import wandb
import torch
import torchmetrics
import torch.optim as optim
from tqdm.auto import tqdm
import torch.nn as nn
import numpy as np

from src.dataset import get_letter_dataset
from src.probes import LinearProbe, MLPProbe
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)

import numpy as np

def all_probe_training_runner(
    # data
    embeddings,
    vocab,
    # task
    alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ",  # Task
    criteria_mode="anywhere",  # "anywhere", "starting" or "posN" (where N is a digit)
    # arch
    probe_type="linear",
    # hyperparameters
    num_epochs=100,  # Define number of training epochs:
    batch_size=32,
    learning_rate=0.001,
    train_test_split=0.8,
    rebalance=False,
    # other config
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    use_wandb=False,
):
    group_name = wandb.util.generate_id() + "_" + alphabet

    # Initialize an empty dictionary to store the learned weights for all letters (or, equivalently, 26 "directions", one for each linear probe)
    embeddings_dim = embeddings.shape[1]
    probes = {letter: LinearProbe(embeddings_dim).to(device) for letter in alphabet}

    # Now loop over the alphabet and train/validate a probe for each letter:

    for i, letter in enumerate(alphabet):
        model = probes[letter]
        nn.init.xavier_uniform_(model.fc.weight)  # best way to do this?

        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10, verbose=True
        )

        if use_wandb:
            config = {
                "letter": letter,
                "criteria_mode": criteria_mode,
                "model_name": "gpt-j",
                "probe_type": probe_type,
                "train_test_split": train_test_split,
                "rebalance": rebalance,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
                "device": device,
            }

            wandb.init(
                project="letter_presence_probes",
                group=group_name,
                config=config,
            )

        train_loader, test_loader = get_letter_dataset(
            criterion=criteria_mode,
            target=letter,
            embeddings=embeddings,
            vocab=vocab,
            batch_size=batch_size,
            rebalance=rebalance,
            test_proportion=1 - train_test_split,
        )

        # Train the probe for the current letter:
        model, other_artifacts = train_letter_probe_runner(
            model=probes[letter],
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            device=device,
            use_wandb=use_wandb,
        )
        
        probes[letter] = model

    if use_wandb:
        # Start a new W&B run for logging the aggregate artifact
        wandb.init(project="letter_presence_probes", name="aggregate_artifact_logging")
        # create_and_log_artifact(
        #     probes,
        #     "all_probe_weights",
        #     "model_tensors",
        #     "All case-insensitive letter presence probe weights tensor",
        # )
        
        # log all probe weights
        # artifacts = []
        # for letter, probe in probes.items():
        #     artifact = wandb.Artifact(
        #         letter,
        #         type="model_tensors",
        #         description=f"Letter presence probe weights tensor for letter {letter}",
        #     )
        #     artifact.add_file(f"probe_{letter}_{criteria_mode}.pt", probe.fc.weight.detach())
        #     artifacts.append(artifact)
        
        # wandb.log_artifact(artifacts)
        
        wandb.finish()  # End the run

    return probes


def train_letter_probe_runner(
    model,
    train_loader,
    test_loader,
    num_epochs,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    use_wandb=False,
):
    loss_fn = nn.BCEWithLogitsLoss()

    steps = 0
    metric_frequency = 10
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        pbar = tqdm(train_loader)

        for batch_embeddings, batch_labels in pbar:
            # Move your data to the chosen device during the training loop and ensure they're float32
            # By explicitly converting to float32, you ensure that the data being fed into your model has the expected data type, and this should resolve the error you en
            # batch_embeddings = batch_embeddings
            # batch_labels = batch_labels.to(device).float()

            optimizer.zero_grad()
            outputs = model(batch_embeddings).squeeze()
            loss = loss_fn(outputs, batch_labels.to(device).float())
            loss.backward()
            optimizer.step()

            if steps % metric_frequency == 0:
                with torch.no_grad():
                    probs = torch.sigmoid(outputs).detach() > 0.5
                    
                    accuracy = torchmetrics.functional.accuracy(probs, batch_labels, task = 'binary', num_classes=2).item()
                    precision = torchmetrics.functional.precision( probs, batch_labels, task= 'binary', num_classes=2).item()
                    recall = torchmetrics.functional.recall(probs, batch_labels, 'binary', num_classes=2).item()
                    f1 = torchmetrics.functional.f1_score(probs, batch_labels, 'binary', num_classes=2).item()
        
                    # predictions
                    pbar.set_description(
                        f"Epoch {epoch+1} / {num_epochs} | Loss: {loss:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}"
                    )
                    if use_wandb:
                        # Log metrics to W&B
                        wandb.log(
                            {
                                "training_metrics/precision": precision,
                                "training_metrics/recall": recall,
                                "training_metrics/loss": loss.item(),
                                # "training_metrics/f1_score": f1_score,
                                "metrics/learning_rate": optimizer.param_groups[0]["lr"],
                            },
                            step=steps,
                        )

            steps += 1
            
        # EVALUATION (VALIDATION) PHASE

        # Set the model to evaluation mode
        model.eval()

        # Keep track of correct predictions and total predictions
        correct_preds = 0
        total_preds = 0
        validation_loss = 0.0

        with torch.no_grad():  # Ensure no gradients are computed during validation
            all_labels = []  # Store all true labels
            all_predictions = []  # Store all model predictions

            for batch_embeddings, batch_labels in test_loader:
                batch_embeddings = batch_embeddings.to(
                    device
                ).float()  # Ensure embeddings are on the correct device and dtype
                batch_labels = batch_labels.to(
                    device
                ).float()  # Ensure labels are on the correct device and dtype

                outputs = model(batch_embeddings).squeeze()

                # Calculate loss on validation data
                loss = loss_fn(outputs, batch_labels)
                validation_loss += loss.item()  # Update validation loss

                # Convert outputs to probabilities
                probs = torch.sigmoid(outputs)
                predictions = (probs > 0.5).float()

                # Update correct and total predictions
                correct_preds += (predictions == batch_labels).sum().item()
                total_preds += batch_labels.size(0)

                # Append batch labels and predictions to all_labels and all_predictions
                all_labels.append(batch_labels.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())

            # Flatten all_labels and all_predictions lists and convert to numpy arrays
            all_labels = np.concatenate(all_labels)
            all_predictions = np.concatenate(all_predictions)

            # Compute F1 Score
            f1 = f1_score(all_labels, all_predictions)

            validation_loss /= len(test_loader)  # Get the average validation loss

            # Calculate accuracy and average loss
            accuracy = correct_preds / total_preds
            precision = precision_score(all_labels, all_predictions)
            recall = recall_score(all_labels, all_predictions)

            # print(f"eval_metrics/accuracy: {accuracy * 100:.2f}%")
            # print(f"eval_metrics/precision: {precision * 100:.2f}%")
            # print(f"eval_metrics/recall: {recall * 100:.2f}%")
            # print(f"eval_metrics/loss: {validation_loss:.4f}")
            # print(f"eval_metrics/f1: {f1:.4f}\n")
            scheduler.step(validation_loss)

            if use_wandb:
                wandb.log(
                    {
                        "eval_metrics/accuracy": accuracy * 100,
                        "eval_metrics/precision": precision * 100,
                        "eval_metrics/recall": recall * 100,
                        "eval_metrics/loss": validation_loss,
                        "eval_metrics/f1": f1,
                    }
                )

    if use_wandb:
        wandb.finish()

    return model, 0
