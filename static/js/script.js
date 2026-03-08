document.getElementById("predictForm").addEventListener("submit", async function (e) {

    e.preventDefault();

    const loading = document.getElementById("loading");
    const resultBox = document.getElementById("result");

    resultBox.style.display = "none";
    loading.style.display = "block";

    const features = [
        parseFloat(document.getElementById("tenure").value),
        parseFloat(document.getElementById("monthlyCharges").value),
        parseFloat(document.getElementById("totalCharges").value),
        parseInt(document.getElementById("contract").value)
    ];

    try {

        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ features: features })
        });

        if (!response.ok) {
            throw new Error("Server error");
        }

        const result = await response.json();

        loading.style.display = "none";
        resultBox.style.display = "block";

        const probability = (result.churn_probability * 100).toFixed(2);

        resultBox.innerHTML = `
            <h3>Prediction: ${result.prediction}</h3>
            <p>Churn Probability: ${probability}%</p>

            <div class="progress">
                <div class="progress-bar bg-danger"
                     role="progressbar"
                     style="width:${probability}%">
                    ${probability}%
                </div>
            </div>
        `;

    } catch (error) {

        loading.style.display = "none";
        resultBox.style.display = "block";
        resultBox.innerHTML = `<p class="text-danger">API connection failed</p>`;

    }

});