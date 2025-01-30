document.getElementById('fareForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the form from refreshing the page

    // Get form values
    const distance = parseFloat(document.getElementById('distance').value);
    const timeOfDay = parseInt(document.getElementById('timeOfDay').value);
    const dayOfWeek = parseInt(document.getElementById('dayOfWeek').value);
    const passengers = parseInt(document.getElementById('passengers').value);
    const traffic = parseInt(document.getElementById('traffic').value);
    const weather = parseInt(document.getElementById('weather').value);
    const baseFare = parseFloat(document.getElementById('baseFare').value);
    const perKmRate = parseFloat(document.getElementById('perKmRate').value);
    const perMinuteRate = parseFloat(document.getElementById('perMinuteRate').value);
    const duration = parseInt(document.getElementById('duration').value);

    // Fare calculation formula
    const trafficFactor = traffic * 2; // Example weight for traffic
    const weatherFactor = weather * 1.5; // Example weight for weather
    const fare = baseFare + (distance * perKmRate) + (duration * perMinuteRate) + trafficFactor + weatherFactor;

    // Display the result
    const outputDiv = document.getElementById('output');
    outputDiv.innerHTML = `<strong>Predicted Fare: $${fare.toFixed(2)}</strong>`;
});
