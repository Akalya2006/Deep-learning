code:
import numpy as np
import matplotlib.pyplot as plt

# Select sample indices from test set
sample_indices = [0, 1, 2, 3, 4]  # You can change these

print("Input Digit Image\tExpected Label\tModel Output\tCorrect (Y/N)")
for idx in sample_indices:
    image = X_test[idx].reshape(1, 28, 28, 1)
    expected_label = np.argmax(y_test[idx])
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)
    correct = "Y" if predicted_label == expected_label else "N"
    
    # Display result
    print(f"Image of {expected_label}\t\t{expected_label}\t\t{predicted_label}\t\t{correct}")
    
    # Optional: Show the image
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"Expected: {expected_label}, Predicted: {predicted_label}, Correct: {correct}")
    plt.axis('off')
    plt.show()
output:
<img width="911" height="278" alt="Screenshot 2025-08-13 112036" src="https://github.com/user-attachments/assets/13c47c80-fa3b-40f9-b438-442d5f3f115b" />
<img width="976" height="284" alt="Screenshot 2025-08-13 112011" src="https://github.com/user-attachments/assets/45fe284d-efe1-47b4-8dbd-47c150e684ae" />
