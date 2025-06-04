import torch.nn as nn                                           # library for neural network operations.
from torch.utils.data import DataLoader, Dataset                # library for data loading and processing.
from collections import Counter                                 # library for counting hashable objects.
import numpy as np                                              # library for numerical operations.
import torch                                                    # main library for tensor operations.
from torch import optim                                         # library for optimization algorithms.
from typing import List, Dict, Tuple
import time                                                     # library for time-related functions.
import os                                                       # library for interacting with the operating system.
import matplotlib.pyplot as plt                                 # library for plotting graphs.

from tqdm import tqdm                                           # library for progress bars.
from sklearn.metrics import precision_recall_fscore_support     # library for computing precision, recall, and F1 score.
from colorama import Fore, Style                                # library for colored terminal text output.


def compute_class_weights(counter: Counter) -> torch.Tensor:
    """
    Compute normalized inverse-frequency class weights for CrossEntropyLoss.

    Parameters
    ----------
    counter : Counter
        A Counter object with class frequencies. Keys should be int class labels (1-indexed).

    Returns
    -------
    torch.Tensor
        A 1D tensor with weights ordered by class index - 1 (i.e. for class labels 1 to N).
    """
    # Ordenar las clases por índice (asumiendo clases 1...N)
    sorted_counts = [counter[i] for i in sorted(counter.keys())]
    counts = np.array(sorted_counts, dtype = np.float32)

    inv_freq = 1.0 / counts
    normalized_weights = inv_freq / inv_freq.sum()

    return torch.tensor(normalized_weights, dtype = torch.float32)


class MultiTaskTrainer:
    def __init__(
        self, model: nn.Module, epochs: int, train_dataset: Dataset, train_dataloader: DataLoader,
        val_dataloader: DataLoader, device: str, lr: float, type2id: dict, town2id: dict,
        lambdas = (5/48, 40/48, 3/48), patience: int = 10, weight_decay: float = 1e-4,
        polarity_counter = None, town_counter = None, type_counter = None,
        checkpoint_path: str = 'trained_model', checkpoint_filename: str = 'checkpoint.pth',
        best_model_filename: str = 'best_model.pth'
    ):
        # Initialize the trainer.
        self.model = model.to(device)
        self.epochs = epochs
        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.patience = patience
        self.best_epoch = 0 
        self.lambdas = lambdas
        self.type2id = type2id
        self.town2id = town2id

        # Calcula pesos con la función
        polarity_weights = compute_class_weights(polarity_counter).to(self.device) # para Polarity
        town_weights = compute_class_weights(town_counter).to(self.device)         # para Town
        type_weights = compute_class_weights(type_counter).to(self.device)         # para Type
        
        # Checkpoint parameters.
        self.checkpoint_path     = checkpoint_path
        self.checkpoint_filename = checkpoint_filename
        self.best_model_filename = best_model_filename

        # Training metadata.
        self.train_loss_history  = []   # Training loss history.
        self.val_loss_history    = []   # Validation loss history.
        self.res_p_history       = []   # Polarity result history.
        self.res_town_history    = []   # Town result history.
        self.res_t_history       = []   # Type result history.
        self.final_score_history = []   # Final score history.
        self.best_final_score    = 0    # Best final score.
        self.n_no_improvement    = 0    # Number of epochs without improvement.
        self.best_state_dict     = None # Store the best model state dict.

        # Optimizer.
        self.optimizer   = optim.Adam(           
            params       = model.parameters(), # parameters to optimize.
            lr           = lr,                 # learning rate.
            weight_decay = weight_decay,       # weight decay.
            betas        = (0.9, 0.999)        # betas for the optimizer.
        )

        # Scheduler.
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer = self.optimizer, # el optimizador al que afecta
            T_0       = 3,              # número de épocas antes del primer reinicio
            T_mult    = 2,              # factor multiplicativo para el intervalo entre reinicios
            eta_min   = 1e-6            # tasa de aprendizaje mínima (no baja de esto)
)
        
        # Define la pérdida
        self.criterion_polarity = nn.CrossEntropyLoss(weight = polarity_weights, ignore_index = -1)
        self.criterion_type = nn.CrossEntropyLoss(weight = type_weights, ignore_index = -1)
        self.criterion_town = nn.CrossEntropyLoss(weight = town_weights, ignore_index = -1)

    def compute_loss(self, polarity_logits: torch.Tensor, town_logits: torch.Tensor, type_logits: torch.Tensor,
                     polarity_targets: torch.Tensor, town_targets: torch.Tensor, type_targets: torch.Tensor,
                     review_mask: torch.Tensor) -> torch.Tensor:
        
        mask = review_mask.view(-1).bool()
        loss_p = self.criterion_polarity(
            polarity_logits.view(-1, polarity_logits.size(-1))[mask],
            polarity_targets.view(-1)[mask]
        )
        loss_town = self.criterion_town(town_logits, town_targets)
        loss_type = self.criterion_type(
            type_logits.view(-1, type_logits.size(-1))[mask],
            type_targets.view(-1)[mask]
        )

        return self.lambdas[0] * loss_p + self.lambdas[1] * loss_town + self.lambdas[2] * loss_type

    def eval_model(self, dataloader: DataLoader) -> dict:
        """
        Evaluates the multitask model.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader for the evaluation dataset.

        Returns
        -------
        dict
            A dictionary containing the evaluation results, including loss and F1 scores.
        """
        self.model.eval()
        all_preds_polarity   = []
        all_targets_polarity = []
        all_preds_town       = []
        all_targets_town     = []
        all_preds_type       = []
        all_targets_type     = []

        losses = []

        with torch.no_grad():
            # Iterate over the batches in the dataloader.
            for input_ids, lengths, polarities, towns, types, review_mask, _ in tqdm(dataloader, desc = 'Evaluation... ', colour = 'blue'):
                input_ids = input_ids.to(self.device)
                lengths = lengths.to(self.device)
                polarities = polarities.to(self.device)
                review_mask = review_mask.to(self.device)

                polarity_logits, town_logits, type_logits = self.model(input_ids, lengths, review_mask)

                town_ids = torch.tensor([self.town2id.get(town, 0) for town in towns], device = self.device)
                type_ids = torch.tensor([[self.type2id.get(t, 0) for t in row] for row in types], device = self.device)

                loss = self.compute_loss(
                    polarity_logits, town_logits, type_logits,
                    (polarities.long() - 1),  # shift 1-5 to 0-4
                    town_ids,
                    type_ids,
                    review_mask
                )
                losses.append(loss.item())

                # Flatten y aplicar máscara
                mask = review_mask.view(-1).bool()

                preds_polarity = polarity_logits.argmax(dim = -1).view(-1)[mask].cpu().tolist()
                targets_polarity = polarities.view(-1)[mask].cpu().tolist()
                
                preds_town = town_logits.argmax(dim = -1).cpu().tolist()
                targets_town = town_ids.cpu().tolist()
                
                preds_type = type_logits.argmax(dim = -1).view(-1)[mask].cpu().tolist()
                targets_type = type_ids.view(-1)[mask].cpu().tolist()

                all_preds_polarity.extend(preds_polarity)
                all_targets_polarity.extend([p - 1 for p in targets_polarity])  # convert back to 0-based

                all_preds_town.extend(preds_town)
                all_targets_town.extend(targets_town)

                all_preds_type.extend(preds_type)
                all_targets_type.extend(targets_type)

        # --- MÉTRICAS ---
        def compute_competition_metrics(y_true: List[int], y_pred: List[int], label_set: List[int]) -> float:
            _, _, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels = label_set, zero_division = 0
            )
            return f1.mean()

        res_p = compute_competition_metrics(all_targets_polarity, all_preds_polarity, list(range(5)))
        res_town = compute_competition_metrics(all_targets_town, all_preds_town, list(self.town2id.values()))
        res_t = compute_competition_metrics(all_targets_type, all_preds_type, list(self.type2id.values()))

        sentiment_score = (2 * res_p + res_t + 3 * res_town) / 6

        return {
            'loss'        : np.mean(losses),
            'res_p'       : res_p,
            'res_t'       : res_t,
            'res_town'    : res_town,
            'final_score' : sentiment_score
        }

    def train(self):
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            loss_train = 0
            epoch_time = time.time() # Start time of the epoch.

            for input_ids, lengths, polarities, towns, types, review_mask, _ in tqdm(self.train_dataloader, desc = 'Training... ', colour = 'green'):
                self.optimizer.zero_grad()

                input_ids = input_ids.to(self.device)
                lengths = lengths.to(self.device)
                polarities = polarities.to(self.device)

                review_mask = review_mask.to(self.device)
                polarity_logits, town_logits, type_logits = self.model(input_ids, lengths, review_mask)

                town_ids = torch.tensor([self.town2id.get(town, 0) for town in towns], device = self.device)
                type_ids = torch.tensor([[self.type2id.get(t, 0) for t in row] for row in types], device = self.device)

                loss = self.compute_loss(
                        polarity_logits, town_logits, type_logits,
                        (polarities.long() - 1),
                        town_ids,
                        type_ids,
                        review_mask
                )
                loss.backward()
                self.optimizer.step()
                loss_train += loss.item()

            avg_loss = loss_train / len(self.train_dataloader)
            val_results = self.eval_model(self.val_dataloader)
            self.scheduler.step(epoch)

            self.train_loss_history.append(avg_loss)
            self.val_loss_history.append(val_results['loss'])
            self.res_p_history.append(val_results['res_p'])
            self.res_town_history.append(val_results['res_town'])
            self.res_t_history.append(val_results['res_t'])
            self.final_score_history.append(val_results['final_score'])

            is_best = val_results['final_score'] > self.best_final_score
            if is_best:
                self.best_final_score = val_results['final_score']
                self.best_state_dict = self.model.state_dict()
                self.n_no_improvement = 0
                self.best_epoch = epoch
                self.best_time = time.time() - start_time
            else:
                self.n_no_improvement += 1

            # Save checkpoint.
            elapsed_time = time.time() - start_time # Elapsed time since the start of training.
            self.save_checkpoint(
                state = {
                    # This epoch information.
                    'epoch'              : epoch,
                    'state_dict'         : self.model.state_dict(),
                    'loss_train'         : avg_loss,
                    'loss_val'           : val_results['loss'],
                    'final_score'        : val_results['final_score'],
                    'res_p'              : val_results['res_p'],
                    'res_town'           : val_results['res_town'],
                    'res_t'              : val_results['res_t'],
                    'elapsed_time'       : elapsed_time,
                    # Best information.
                    'best_final_score'   : self.best_final_score,
                    'best_epoch'         : self.best_epoch,
                    'best_time'          : self.best_time,
                    # History information.
                    'train_loss_history' : self.train_loss_history,
                    'val_loss_history'   : self.val_loss_history,
                    'res_p_history'      : self.res_p_history,
                    'res_town_history'   : self.res_town_history,
                    'res_t_history'      : self.res_t_history,
                    'final_score_history': self.final_score_history,
                    # Optimizer and scheduler state.
                    'optimizer'          : self.optimizer.state_dict(),
                    'scheduler'          : self.scheduler.state_dict(),
                },
                is_best = is_best
            )

            # Early stopping.
            if self.n_no_improvement >= self.patience:
                tqdm.write(f"{Fore.RED}Early stopping triggered at epoch {epoch}!{Style.RESET_ALL}")
                break
            elif loss_train < 1e-5:
                tqdm.write(f"{Fore.RED}Loss is too low, stopping training!{Style.RESET_ALL}")
                break

            tqdm.write(f"{Style.BRIGHT}{Fore.CYAN} Epoch{epoch:>3}/{self.epochs}{Style.RESET_ALL}"
                f"  ➤ Training loss: {Fore.GREEN}{avg_loss:.6f}{Style.RESET_ALL} | "
                f"Val Loss: {Fore.YELLOW}{val_results['loss']:.4f}{Style.RESET_ALL} | "
                f"Res_P = {Fore.MAGENTA}{val_results['res_p']:.4f}{Style.RESET_ALL} | "
                f"Res_T = {Fore.MAGENTA}{val_results['res_t']:.4f}{Style.RESET_ALL} | "
                f"Res_Town = {Fore.MAGENTA}{val_results['res_town']:.4f}{Style.RESET_ALL} | "
                f"Final Score = {Fore.MAGENTA}{val_results['final_score']:.4f}{Style.RESET_ALL} | "
                f"Epoch Time: {time.time() - epoch_time:.2f} s | "
                f"Total Time: {elapsed_time:.2f} s"
            )
        print(f"\n{Style.BRIGHT}--- Total Training Time: {elapsed_time:.2f} seconds ---{Style.RESET_ALL}")
        print(f"Best model achieved at epoch {self.best_epoch} with Final Score = {self.best_final_score:.4f}")

    def save_checkpoint(self, state: dict, is_best: bool) -> None:
        """
        Save a checkpoint of the model, including the state dictionary, training metadata,
        optimizer state, and the initialization parameters used to construct the model.

        Parameters
        ----------
        state : dict
            Dictionary with the model state, best model state, loss histories, and optimizer state.
        is_best : bool
            If True, also save the model as the best checkpoint.
        """
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path, exist_ok = True)

        if is_best:
            state['best_model_state_dict'] = self.model.state_dict()

        filepath = os.path.join(self.checkpoint_path, self.checkpoint_filename)
        torch.save(state, filepath)

        if is_best:
            best_path = os.path.join(self.checkpoint_path, self.best_model_filename)
            torch.save({'best_model_state_dict': self.model.state_dict()}, best_path)

    def plot_metrics(self):
        epochs = list(range(1, len(self.final_score_history) + 1))
    
        fig, ax = plt.subplots(figsize = (10,6))
        ax.plot(epochs, self.res_p_history, marker = 'o', label = r'$Res_P$ (Polarity)', color = '#1f77b4')
        ax.plot(epochs, self.res_t_history, marker = 's', label = r'$Res_T$ (Type)', color = '#ff7f0e')
        ax.plot(epochs, self.res_town_history, marker = '^', label = r'$Res_{Town}$ (Magical Towns)', color = '#2ca02c')
        ax.plot(epochs, self.final_score_history, marker = '*', linestyle = '--', linewidth = 2.5, label = 'Final Score', color = '#d62728')
        ax.axvline(self.best_epoch, linestyle = "--", color = "gray", alpha = 0.7, label = f"Best Epoch: {self.best_epoch}")    
        ax.set_title("Metric Evolution per Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()

    def load_model(self, filename: str = 'best_model.pth', load_best: bool = True):
        path = os.path.join(self.checkpoint_path, filename)
        checkpoint = torch.load(path, map_location=self.device)
    
        if load_best and 'best_model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['best_model_state_dict'])
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            raise ValueError(f"No se encontró un modelo válido en {filename}.")
    
        self.model.to(self.device)
        self.model.eval()
        print(f"Modelo cargado desde {filename}")