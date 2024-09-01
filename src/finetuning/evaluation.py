import torch
def evaluate(model, dataloader):
    model.eval()  # Set model to evaluation mode
    total_eval_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch[0].to('cuda')
            attention_masks = batch[1].to('cuda')
            labels = batch[2].to('cuda')

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                labels=labels
            )

            loss = outputs.loss
            total_eval_loss += loss.item()

            # Get predictions
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(predictions == labels).item()

    avg_eval_loss = total_eval_loss / len(dataloader)
    accuracy = correct_predictions / len(dataloader.dataset)
    print(f"Validation Loss: {avg_eval_loss}")
    print(f"Validation Accuracy: {accuracy}")
