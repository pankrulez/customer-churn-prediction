from src.model_training import train_pipeline, save_pipeline

if __name__ == "__main__":
    pipeline, metrics = train_pipeline()
    save_pipeline(pipeline)

    print("Model trained and saved.")
    print("Metrics:", metrics)