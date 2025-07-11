// static/script.js

const form = document.getElementById("predict-form");
const userInput = document.getElementById("user-input");
const chatbox = document.getElementById("chatbox");
const loading = document.getElementById("loading");
const resultArea = document.getElementById("result-area");
const diseaseSpan = document.getElementById("predicted-disease");
const adviceSpan = document.getElementById("advice");

let chart;

form.addEventListener("submit", async function (e) {
  e.preventDefault();
  const text = userInput.value.trim();
  if (!text) return;

  appendUserMessage(text);
  loading.classList.remove("hidden");
  resultArea.classList.add("hidden");
  userInput.value = "";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_input: text })
    });

    const data = await response.json();
    console.log("Received:", data); // ✅ Debug output to verify disease name

    if (data.error) {
      appendBotMessage("❌ Something went wrong. Try again.");
      return;
    }

    // ✅ Display predicted disease and advice
    appendBotMessage(`🧠 I predict: ${data.predicted_disease}`);
    diseaseSpan.textContent = data.predicted_disease;
    adviceSpan.textContent = data.advice;
    resultArea.classList.remove("hidden");

    drawPieChart(data.probabilities);
  } catch (error) {
    appendBotMessage("❌ Something went wrong. Try again.");
    console.error("Prediction error:", error);
  } finally {
    loading.classList.add("hidden");
  }
});

function appendUserMessage(msg) {
  const div = document.createElement("div");
  div.className = "user-msg";
  div.textContent = "🧑 " + msg;
  chatbox.appendChild(div);
  chatbox.scrollTop = chatbox.scrollHeight;
}

function appendBotMessage(msg) {
  const div = document.createElement("div");
  div.className = "bot-msg";
  div.textContent = msg;
  chatbox.appendChild(div);
  chatbox.scrollTop = chatbox.scrollHeight;
}

function startVoiceRecognition() {
  const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
  recognition.lang = "en-US";
  recognition.start();

  recognition.onresult = function (event) {
    const transcript = event.results[0][0].transcript;
    userInput.value = transcript;
  };

  recognition.onerror = function (event) {
    alert("Voice input error: " + event.error);
  };
}

function drawPieChart(probabilities) {
  const labels = Object.keys(probabilities);
  const values = Object.values(probabilities);

  const ctx = document.getElementById("pieChart").getContext("2d");
  if (chart) chart.destroy();

  chart = new Chart(ctx, {
    type: "pie",
    data: {
      labels: labels,
      datasets: [{
        label: "Prediction Probabilities",
        data: values,
        backgroundColor: [
          "#29b6f6", "#66bb6a", "#ef5350", "#ffa726", "#ab47bc",
          "#26c6da", "#d4e157", "#8d6e63", "#ec407a", "#78909c"
        ],
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          position: "bottom",
        },
      },
    },
  });
}
