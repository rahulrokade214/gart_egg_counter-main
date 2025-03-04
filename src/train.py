import tensorflow as tf
from data_preparation import prepare_dataset, split_dataset, augment_data
from model import create_model

def train_model(data_dir, epochs=50, batch_size=32):
    # Prepare data
    images, labels = prepare_dataset(data_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(images, labels)
    
    # Create and compile model
    model = create_model(input_shape=X_train.shape[1:])
    
    # Data augmentation
    data_augmentation = tf.keras.Sequential([
        layers.Lambda(lambda x: tf.numpy_function(augment_data, [x], tf.float32))
    ])
    
    # Create dataset objects
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size)
    train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
    
    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
    )
    
    # Evaluate the model
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test MAE: {test_mae}")
    
    # Save the model
    model.save('egg_counter_model.h5')
    
    return history

if __name__ == "__main__":
    data_dir = "path/to/your/image/dataset"
    history = train_model(data_dir)
    
    # Plot training history
    import matplotlib.pyplot as plt
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig('training_history.png')