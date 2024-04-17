import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';
import { createRoot } from 'react-dom';

// const root
createRoot = ReactDOM.createRoot(document.getElementById('root'));
createRoot.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);