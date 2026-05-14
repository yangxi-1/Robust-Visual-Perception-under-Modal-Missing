from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    pbar = tqdm(loader, desc="Training", leave=False)

    for batch in pbar:
        image = batch["image"].to(device)
        text_inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device)
        }
        mask = batch["mask"].to(device)
        label = batch["label"].to(device)

        output = model(image, text_inputs, mask)

        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 🔥 实时更新进度条信息
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}"
        })

    return total_loss / len(loader)
