// import React from "react";
// import ReactDOM from "react-dom/client";
// import App from "./App.js";

// // Render the App component into the root div
// const root = ReactDOM.createRoot(document.getElementById("root"));
// root.render(<App />);
// import React from "react";
import "./styles.css"; // optional CSS file

function App() {
  return (
    <div className="container">
      <h1>Hello, React!</h1>
      <p>This is my first React web page.</p>
      <button onClick={() => alert("You clicked me!")}>
        Click Me
      </button>
    </div>
  );
}

export default App;