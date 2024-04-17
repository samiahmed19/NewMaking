import React from 'react';
import { Link } from 'react-router-dom';

const NavBar = ({ isOpen, onClose }) => {
    const navStyle = {
        height: '100%',
        width: isOpen ? '250px' : '0',
        position: 'fixed',
        zIndex: 1,
        top: 0,
        left: 0,
        backgroundColor: '#232323',
        transition: 'width 0.5s',
        overflowX: 'hidden',
        paddingTop: '20px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center'
    };

    const linkStyle = {
        color: '#fff',
        textDecoration: 'none',
        margin: '10px 0',
    };

    const closeButtonStyle = {
        backgroundColor: 'transparent',
        color: '#fff',
        border: 'none',
        cursor: 'pointer',
        marginTop: '20px',
    };

    return (
        <div className="sidenav" style={navStyle}>
            {isOpen && (
                <>
                    <Link to="/" style={linkStyle}>Home</Link>
                    <Link to="/prediction" style={linkStyle}>Prediction</Link>
                    <Link to="/login" style={linkStyle}>Login</Link>
                    <button className="close-sidebar-btn" style={closeButtonStyle} onClick={onClose}>Close Sidebar</button>
                </>
            )}
        </div>
    );
};

export default NavBar;
