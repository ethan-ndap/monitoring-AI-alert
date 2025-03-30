import React, { useState } from "react"; // Import React
import axios from "axios";

const AnomalyAlert = () => {
  const [anomaly, setAnomaly] = useState("");
  const [phoneNumber, setPhoneNumber] = useState("");

  const sendAlert = async () => {
    if (!anomaly || !phoneNumber) {
      alert("Enter anomaly description and phone number.");
      return;
    }

    try {
      await axios.post("http://localhost:5000/send-alert", {
        message: `Alert! Abnormal behavior detected: ${anomaly}`,
        phoneNumber,
      });
      alert("Alert sent successfully!");
    } catch (error) {
      alert("Failed to send alert: " + error.message);
    }
  };

  return (
    <div className="p-4 border rounded-lg shadow-md max-w-md">
      <h2 className="text-xl font-bold mb-2">Anomaly Alert</h2>
      <input
        type="text"
        placeholder="Describe anomaly..."
        value={anomaly}
        onChange={(e) => setAnomaly(e.target.value)}
        className="w-full p-2 border rounded mb-2"
      />
      <input
        type="tel"
        placeholder="Enter phone number"
        value={phoneNumber}
        onChange={(e) => setPhoneNumber(e.target.value)}
        className="w-full p-2 border rounded mb-2"
      />
      <button
        onClick={sendAlert}
        className="bg-red-500 text-white p-2 rounded w-full"
      >
        Send Alert
      </button>
    </div>
  );
};

export default AnomalyAlert;
