"""Training configuration"""

class TrainingConfig:
    """Training hyperparameters"""
    
    # Training duration
    NUM_EPOCHS = 2
    
    # Batch sizes
    BATCH_SIZE_TRAIN = 1
    BATCH_SIZE_EVAL = 1
    GRADIENT_ACCUMULATION_STEPS = 4
    
    # Learning
    LEARNING_RATE = 2e-4
    WARMUP_STEPS = 10
    
    # Optimization
    FP16 = True
    GRADIENT_CHECKPOINTING = False  # Disable for GPU compatibility (CUDA/cuBLAS issues)
    OPTIMIZER = "paged_adamw_8bit"
    
    # Logging
    LOGGING_STEPS = 10
    
    # Saving
    SAVE_STEPS = 100
    SAVE_TOTAL_LIMIT = 1
    
    # Evaluation
    EVAL_STRATEGY = "steps"
    EVAL_STEPS = 100