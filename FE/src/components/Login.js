import React, { useState } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import './login.css'; 
import NavBar2 from './NavBar2'; // Changed NavBar2 to NavBar

function Login() {
  const [formData, setFormData] = useState({ email: '', password: '' });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await axios.post('http://localhost:5000/login', formData);
      console.log('Login successful');
      // Redirect to prediction page after login
      window.location.href = '/prediction';
    } catch (error) {
      console.error('Login failed:', error);
    }
  };

  return (
    <div className="login-container">
      <NavBar2 />
      <div className="login-box">
        <h2 className="login-title">Login</h2>
        <form className="login-form" onSubmit={handleSubmit}>
          <input className="login-input" type="email" name="email" placeholder="Email" value={formData.email} onChange={handleChange} />
          <input className="login-input" type="password" name="password" placeholder="Password" value={formData.password} onChange={handleChange} />
          <button className="login-button" type="submit">Login</button>
        </form>
        <div className="login-link">
          <Link to="/">Signup</Link>
        </div>
      </div>
    </div>
  );
}

export default Login;
