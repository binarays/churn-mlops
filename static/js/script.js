function yesNoToInt(val) {
    return val === "Yes" ? 1 : 0;
}

function genderToInt(val) {
    return val === "Male" ? 1 : 0;
}
document.getElementById("predictForm").addEventListener("submit", async function (e) {

    e.preventDefault();

    const loading = document.getElementById("loading");
    const resultBox = document.getElementById("result");
    const predictlable = document.getElementById("prediction_lable");

    resultBox.style.display = "none";
    loading.style.display = "block";

    const features = [
        genderToInt(document.getElementById("gender").value),
        parseInt(document.getElementById("seniorCitizen").value),
        yesNoToInt(document.getElementById("partner").value),
        yesNoToInt(document.getElementById("dependents").value),
        parseFloat(document.getElementById("tenure").value),
        yesNoToInt(document.getElementById("phoneService").value),
        yesNoToInt(document.getElementById("multipleLines").value),

        document.getElementById("internetService").value === "DSL" ? 1 : 0,
        document.getElementById("internetService").value === "Fiber optic" ? 1 : 0,
        document.getElementById("internetService").value === "No" ? 1 : 0,

        yesNoToInt(document.getElementById("onlineSecurity").value),
        yesNoToInt(document.getElementById("onlineBackup").value),

        document.getElementById("deviceProtection").value === "Yes" ? 1 : 0,
        document.getElementById("deviceProtection").value === "No" ? 1 : 0,

        document.getElementById("techSupport").value === "Yes" ? 1 : 0,
        document.getElementById("techSupport").value === "No" ? 1 : 0,

        document.getElementById("streamingTV").value === "Yes" ? 1 : 0,
        document.getElementById("streamingTV").value === "No" ? 1 : 0,

        document.getElementById("streamingMovies").value === "Yes" ? 1 : 0,
        document.getElementById("streamingMovies").value === "No" ? 1 : 0,

        document.getElementById("contract").value === "Month-to-month" ? 1 : 0,
        document.getElementById("contract").value === "One year" ? 1 : 0,
        document.getElementById("contract").value === "Two year" ? 1 : 0,

        yesNoToInt(document.getElementById("paperlessBilling").value),

        document.getElementById("paymentMethod").value === "Electronic check" ? 1 : 0,
        document.getElementById("paymentMethod").value === "Mailed check" ? 1 : 0,
        document.getElementById("paymentMethod").value === "Bank transfer (automatic)" ? 1 : 0,
        document.getElementById("paymentMethod").value === "Credit card (automatic)" ? 1 : 0,

        parseFloat(document.getElementById("monthlyCharges").value),
        parseFloat(document.getElementById("totalCharges").value)
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