if RUN_EXPERIMENT:
    train_and_score(fresh_trainer(num_train_epochs=3), "3 epochs", each_epoch=True)
