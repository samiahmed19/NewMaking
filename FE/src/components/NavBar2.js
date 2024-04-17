import React from 'react';
import { Link } from 'react-router-dom';
import './nav2.css';

const NavBar2 = () => {
    return (
        <nav className="navbar">
            <div className="logo">Stock Price Prediction Using Machine Learning & Deep Learning Methods</div>
            <ul className="nav-links">
                <li><Link to="/" className="nav-link">Home</Link></li>
                <li><Link to="/login" className="nav-link">Login</Link></li>
            </ul>
        </nav>
    );
};

export default NavBar2;
