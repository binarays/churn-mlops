function clearFields() {
    document.getElementById("tenure").value = "";
    document.getElementById("monthlyCharges").value = "";
    document.getElementById("totalCharges").value = "";
    document.getElementById("contract").selectedIndex = 0;

    document.getElementById("result").innerHTML = "";
}