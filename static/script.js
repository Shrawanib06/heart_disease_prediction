function submitForm() {
    // Select all inputs inside the #predictionForm div
    const inputs = document.querySelectorAll('#predictionForm input, #predictionForm select');

    // Create a data object and populate it with key-value pairs
    const data = {};
    let isValid = true; // Flag to check if all required inputs have values

    inputs.forEach(input => {
        // Trim to remove extra whitespace
        const value = input.value.trim();

        // Check if input is empty
        if (!value) {
            isValid = false;

            // Optionally highlight the empty input fields (optional)
            input.style.border = '2px solid red'; 
        } else {
            input.style.border = ''; // Reset border if valid
        }

        data[input.name] = value;
    });

    // If any required input is empty, display a message and stop the function
    if (!isValid) {
        document.getElementById('result').innerHTML = '<h3>Please fill in all the required fields.</h3>';
        return;
    }

    // Debugging: Check if the data is collected correctly
    console.log('Collected data:', data);

    // Send the data to the server using a POST request
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
    })
    .then(response => response.json())
    .then(data => {
        console.log('Response from server:', data);

        // Check the prediction result and display the message
        const prediction = data.prediction;
        const resultMessage = prediction === 0 
            ? 'You do not have heart disease.'
            : 'You may have heart disease. Please consult a doctor for further evaluation.';
        
        // Display the result message in the result div
        document.getElementById('result').innerHTML = `<h3>${resultMessage}</h3>`;
    })
    .catch(error => {
        console.error('Error:', error);
    });
}
