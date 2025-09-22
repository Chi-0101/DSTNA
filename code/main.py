from DSTNA import DSTNA
from dataloader import get_dataset
import torch.nn.functional as F
from torch.optim import SGD
from args import arg_parser
import torch
from tqdm import tqdm
import json
import os
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

REFRESH_FREQUENCY = 10


def plot_performance(accuracy_history, loss_history):
    """Plot accuracy and loss curves."""
    iterations = list(range(len(accuracy_history)))

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(iterations, [acc * 100 for acc in accuracy_history], 'b-', linewidth=2)
    plt.title('Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Samples', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 100)

    plt.subplot(1, 2, 2)
    plt.plot(iterations, loss_history, 'r-', linewidth=2)
    plt.title('Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Samples', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


def save_results(data, suffix=None):
    """Save results to JSON file and plot performance."""
    accuracy_history = data.get('accuracy_history', [])
    loss_history = data.get('loss_history', [])

    plot_performance(accuracy_history, loss_history)

    try:
        root_path = data['config'].get('root', './results')
        dataset = data['config'].get('dataset', 'default_dataset')
        modelname = data['config'].get('model', 'DSTNA')
        if suffix:
            modelname = modelname + suffix
        filename = data['config'].get('filename', 'results.json')

        path = os.path.join(root_path, dataset, modelname)
        os.makedirs(path, exist_ok=True)

        json_path = os.path.join(path, filename)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"Results saved to: {json_path}")
    except Exception as e:
        print(f"Warning: Could not save results: {e}")

    print(f"\nFinal Accuracy: {data.get('final_accuracy', 0):.2f}%")
    print(f"Final Loss: {data.get('final_loss', 0):.4f}")


def get_pred_label(pred, weights):
    """Get predicted label from ensemble prediction."""
    weighted_preds = weights.unsqueeze(-1).unsqueeze(-1) * pred
    ensemble_pred = torch.sum(weighted_preds, dim=0)
    return ensemble_pred.argmax().item()


def torch_train(optimizer, pred, label, weights):
    """Train the model for one step."""
    optimizer.zero_grad()
    pred_ = pred.detach()

    label_val = label.item() if hasattr(label, 'item') else int(label)

    # Calculate losses for each layer
    losses = []
    for p in pred_:
        p_input = p.unsqueeze(0) if p.dim() == 1 else p
        label_tensor = torch.tensor([label_val], dtype=torch.long, device=p_input.device)
        loss = F.cross_entropy(p_input, label_tensor)
        losses.append(loss)
    losses = torch.stack(losses)

    # Calculate final weighted loss
    weighted_preds = weights.unsqueeze(-1).unsqueeze(-1) * pred
    update_pred = torch.sum(weighted_preds, dim=0)
    final_label = torch.tensor([label_val], dtype=torch.long, device=update_pred.device)
    loss = F.cross_entropy(update_pred, final_label)

    loss.backward()
    optimizer.step()

    return loss.item(), losses


def main():
    args = arg_parser.parse_args()
    args.model = 'DSTNA'

    print(args)

    # Load dataset
    try:
        dataloader, dataset = get_dataset(args)
        args.nIn = dataset.get_nIn()
        args.nOut = dataset.get_nOut()
        args.total = dataset.__len__()
        print(f"Dataset: {args.dataset}, Input: {args.nIn}, Output: {args.nOut}, Samples: {args.total}")
    except Exception as e:
        print(f'Error loading dataset: {e}')
        return

    # Initialize model
    model = DSTNA(args)
    optimizer = SGD(model.parameters(), lr=args.lr)
    print(f"Model: DSTNA, Optimizer: SGD(lr={args.lr})")

    # Training setup
    pbar = tqdm(total=args.total, ncols=100)
    data_output = {
        'config': args.__dict__,
        'pred': [],
        'label': []
    }

    correct_predictions = 0
    total_samples = 0
    accuracy_history = []
    loss_history = []
    all_predictions = []
    all_labels = []

    # Training loop
    try:
        for i, (x, y) in enumerate(dataloader):
            if hasattr(y, 'dim') and y.dim() > 0 and y.size(0) == 1:
                y = y.squeeze(0)

            # Predict before training
            with torch.no_grad():
                pred_before_training = model(x)

            pl = get_pred_label(pred_before_training, model.alpha_vector)
            y_scalar = y.item() if isinstance(y, torch.Tensor) else y

            if pl == y_scalar:
                correct_predictions += 1
            total_samples += 1
            current_accuracy = correct_predictions / total_samples

            all_predictions.append(pl)
            all_labels.append(y_scalar)

            # Train
            pred_for_training = model(x)
            loss, losses = torch_train(optimizer, pred_for_training, y, weights=model.gamma_vector)

            # Update memory pools
            if hasattr(model, 'update_local_memory'):
                model.update_local_memory(x, y)
                model.update_global_memory(x, y, loss)

                if hasattr(model, 'update_window_samples'):
                    model.update_window_samples(pred_before_training.detach(), y)

                memory_train_freq = getattr(model, 'memory_train_freq', 50)
                if i > 0 and i % memory_train_freq == 0 and hasattr(model, 'train_with_memory'):
                    model.train_with_memory(optimizer)

            # Update model weights
            model.update_weights(losses)

            # Save predictions
            data_output['pred'].append(pl)
            data_output['label'].append(y_scalar)
            accuracy_history.append(current_accuracy)
            loss_history.append(loss)

            # Update progress bar
            if i % REFRESH_FREQUENCY == 0:
                pbar.set_description('Loss:{:.4f}, Acc:{:.4f}'.format(loss, current_accuracy))
                pbar.update(REFRESH_FREQUENCY)

        # Final results
        final_accuracy = correct_predictions / total_samples
        data_output['final_accuracy'] = final_accuracy * 100
        data_output['accuracy_history'] = accuracy_history
        data_output['loss_history'] = loss_history
        data_output['final_loss'] = np.mean(loss_history[-100:]) if len(loss_history) >= 100 else np.mean(loss_history)

        pbar.update(args.total % REFRESH_FREQUENCY)
        pbar.close()

        print(f"\nTraining completed!")
        print(f"Final Accuracy: {final_accuracy * 100:.2f}%")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return

    # Save results
    suffix = None
    if args.del_all:
        suffix = "_DEL_ALL"
    elif args.del_da:
        suffix = "_DEL_DA"
    elif args.del_pa:
        suffix = "_DEL_PA"

    save_results(data_output, suffix)

    print("\n" + "=" * 50)
    print("Training Summary")
    print("=" * 50)
    print(f"Total Samples: {total_samples}")
    print(f"Final Accuracy: {final_accuracy * 100:.2f}%")
    print(f"Final Loss: {data_output['final_loss']:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    main()