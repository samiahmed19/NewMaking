import React, { useState } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import './signup.css';
import NavBar2 from './NavBar2';

function Signup() {
  const [formData, setFormData] = useState({ username: '', email: '', password: '', phoneNumber: '' });

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await axios.post('http://localhost:5000/signup', formData);
      console.log('User signed up successfully');
      // Redirect to login page after signup
      window.location.href = '/login';
    } catch (error) {
      console.error('Signup failed:', error);
    }
  };

  return (
    <div>
      <NavBar2 />
      <div className="container">
        <h2 className="heading">Signup</h2>
        <form onSubmit={handleSubmit} className="form">
          <div className="form-group">
            <input type="text" name="username" placeholder="Username" value={formData.username} onChange={handleChange} className="input" />
          </div>
          <div className="form-group">
            <input type="email" name="email" placeholder="Email" value={formData.email} onChange={handleChange} className="input" />
          </div>
          <div className="form-group">
            <input type="password" name="password" placeholder="Password" value={formData.password} onChange={handleChange} className="input" />
          </div>
          <div className="form-group">
            <input type="text" name="phoneNumber" placeholder="Phone Number" value={formData.phoneNumber} onChange={handleChange} className="input" />
          </div>
          <button type="submit" className="button">Signup</button>
        </form>
        <Link to="/login" className="link">Login</Link>
      </div>
    </div>
  );
}

export default Signup;
