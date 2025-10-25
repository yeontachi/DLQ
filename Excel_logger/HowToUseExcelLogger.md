## excel_logger.py 사용법 (테스트 결과가 엑셀 자동화 프로그램)


`pip install pandas openpyxl psutil`

Pytorch 학습 루프 적용 예시
```python
from excel_logger import ExcelLogger 
logger = ExcelLogger(excel_path="results.xlsx", tag = "ViT_2x2_CIFAR10") ##

def main(cfg: Config):
    set_seed(cfg.seed)
    train_loader, test_loader = get_dataloaders(cfg)

    model = ViT(cfg).to(cfg.device)
    print(f"Model params: {count_params(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device.startswith("cuda")))

    best_acc = 0.0
    for epoch in range(cfg.epochs):
        lr = cosine_warmup_lr(epoch, cfg.lr, cfg.epochs, cfg.warmup_epochs)
        for g in optimizer.param_groups:
            g["lr"] = lr

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, cfg.device, scaler)
        val_loss, val_acc = evaluate(model, test_loader, cfg.device)
        
        # 하이퍼 파라미터/노트와 함께 기록
        logger.log(
            metrics={"train_loss":train_loss, "train_acc":train_acc,
                    "val_loss":val_loss, "val_acc":val_acc, "epoch":epoch+1},
            params={"model":"ViT-B(mini)", "patch":2, "lr":lr, "epochs":cfg.epochs},
            extra={"note":"Patch Size 2x2"}
        )
```