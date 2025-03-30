import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App"; // Import your App component

const rootElement = document.getElementById("app") as HTMLElement;
const root = ReactDOM.createRoot(rootElement);

root.render(
  <React.StrictMode>
    <App /> {/* Render your App component here */}
  </React.StrictMode>
);
