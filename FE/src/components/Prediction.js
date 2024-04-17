import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './prediction.css'; // Import CSS file for styling
import NavBar from './Navbar';

const Prediction = () => {
    const [stockSymbol, setStockSymbol] = useState('');
    const [predictionResult, setPredictionResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [expandedImage, setExpandedImage] = useState(null);
    // const [cancelToken, setCancelToken] = useState(null);
    const [sidebarOpen, setSidebarOpen] = useState(false); // State variable for sidebar
    const sidebarRef = useRef(null); // Ref for sidebar
    // const [selectedAlgorithm, setSelectedAlgorithm] = useState('lstm');

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (sidebarRef.current && !sidebarRef.current.contains(event.target)) {
                setSidebarOpen(false);
            }
        };

        document.addEventListener('click', handleClickOutside);

        return () => {
            document.removeEventListener('click', handleClickOutside);
        };
    }, [sidebarRef]);



    const handleInputChange = (event) => {
        setStockSymbol(event.target.value);
    };

    const handlePrediction = async () => {
        setLoading(true);
        setError(null);
        const source = axios.CancelToken.source();
        // setCancelToken(source);

        try {
            if (!stockSymbol) {
                setError('Please enter a ticker symbol.');
                setLoading(false);
                return;
            }

            const response = await axios.post(
                'http://localhost:5000/predict',
                {
                    nm: stockSymbol,
                    // algorithm: selectedAlgorithm // Pass selected algorithm as a parameter
                },
                { cancelToken: source.token }
            );

            if (response.data.error) {
                if (response.data.error.includes('No timezone found')) {
                    setError('Stock Symbol (Ticker) Not Found. Please Enter a Valid Ticker Symbol.');
                } else {
                    setError(response.data.error);
                }
                setPredictionResult(null);
            } else {
                setPredictionResult(response.data);
                // Store prediction results in sessionStorage
                sessionStorage.setItem('predictionResult', JSON.stringify(response.data));
            }
        } catch (error) {
            if (axios.isCancel(error)) {
                console.log('Request canceled:', error.message);
            } else {
                console.error('Prediction failed:', error);
                setError('Failed to fetch predictions. Please try again.');
            }
        } finally {
            setLoading(false);
        }
    };


    const handleClearResults = () => {
        // Clear prediction results stored in sessionStorage
        sessionStorage.removeItem('predictionResult');
        setPredictionResult(null);
    };

    const handleImageClick = (imageUrl) => {
        setExpandedImage(imageUrl);
    };

    const handleCloseExpandedImage = () => {
        setExpandedImage(null);
    };

    // const handleStopPrediction = () => {
    //     if (cancelToken) {
    //         cancelToken.cancel('Operation canceled by the user.'); // Cancel the ongoing request
    //     }
    //     setLoading(false);
    //     setError('Prediction stopped by the user.');
    // };

    const toggleSidebar = () => {
        setSidebarOpen(!sidebarOpen);
    };
    const handleCloseSidebar = () => {
        setSidebarOpen(false);
    };
    // const handleAlgorithmChange = (event) => {
    //     setSelectedAlgorithm(event.target.value);
    // };

    return (
        <div className="prediction-container">
            <NavBar isOpen={sidebarOpen} onClose={handleCloseSidebar} />

            <div className="main-content">
                <button className="toggle-sidebar-btn" onClick={toggleSidebar}>
                    {sidebarOpen ? (
                        '\u2715'
                    ) : (
                        <svg
                            xmlns="http://www.w3.org/2000/svg"
                            viewBox="0 0 24 24"
                            width="24"
                            height="24"
                            fill="currentColor"
                        >
                            <path d="M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z" />
                        </svg>
                    )}
                </button>

                <div className="input-bar">
                    <input
                        type="text"
                        value={stockSymbol}
                        onChange={handleInputChange}
                        placeholder="Enter Stock Symbol"
                    />
                    <button onClick={handlePrediction} disabled={loading}>Predict</button>
                    <button onClick={handleClearResults}>Clear Results</button>
                    {/* {loading && <button onClick={handleStopPrediction}>Stop</button>} */}
                </div>

                <div>Todays {console.log(stockSymbol)} Stock Information</div>
                <div className='trend-image'>
                    {predictionResult && predictionResult.t_img && (
                        <div className='trends-tag'>
                            {/* <span>TRENDS Prediction</span> */}
                            <img
                                src={predictionResult.t_img}
                                alt="TRENDS Prediction"
                                className={expandedImage === predictionResult.t_img ? 'expanded-image' : ''}
                                onClick={() => handleImageClick(predictionResult.t_img)}
                            />
                        </div>
                    )}
                </div>


                {loading && <p className="loading">Loading predictions...</p>}
                {error && <p className="error">{error}</p>}

                {predictionResult && (
                    <div className="result-container">
                        <div className="stock-info">
                            <h3>Stock Information</h3>
                            <div className="info-box">
                                <span className="info-label">Open:</span>
                                <span className="info-value">{predictionResult.open_s}</span>
                            </div>
                            <div className="info-box">
                                <span className="info-label">Close:</span>
                                <span className="info-value">{predictionResult.close_s}</span>
                            </div>
                            <div className="info-box">
                                <span className="info-label">Adj Close:</span>
                                <span className="info-value">{predictionResult.adj_close}</span>
                            </div>
                            <div className="info-box">
                                <span className="info-label">High:</span>
                                <span className="info-value">{predictionResult.high_s}</span>
                            </div>
                            <div className="info-box">
                                <span className="info-label">Low:</span>
                                <span className="info-value">{predictionResult.low_s}</span>
                            </div>
                            <div className="info-box">
                                <span className="info-label">Volume:</span>
                                <span className="info-value">{predictionResult.vol}</span>
                            </div>
                        </div>


                        <div className="prediction-images">

                            <div>
                                <img
                                    src={predictionResult.lstm_img}
                                    alt="LSTM Prediction"
                                    className={expandedImage === predictionResult.lstm_img ? 'expanded-image' : ''}
                                    onClick={() => handleImageClick(predictionResult.lstm_img)}
                                />
                                <span>LSTM Prediction</span>
                            </div>

                            <div>
                                <img
                                    src={predictionResult.bi_img}
                                    alt="Bi-LSTM Prediction"
                                    className={expandedImage === predictionResult.bi_img ? 'expanded-image' : ''}
                                    onClick={() => handleImageClick(predictionResult.bi_img)}
                                />
                                <span>Bi-LSTM Prediction</span>
                            </div>

                            <div>
                                <img
                                    src={predictionResult.gru_img}
                                    alt="GRU Prediction"
                                    className={expandedImage === predictionResult.gru_img ? 'expanded-image' : ''}
                                    onClick={() => handleImageClick(predictionResult.gru_img)}
                                />
                                <span>GRU Prediction</span>
                            </div>

                            <div>
                                <img
                                    src={predictionResult.mlp_img}
                                    alt="MLP Prediction"
                                    className={expandedImage === predictionResult.mlp_img ? 'expanded-image' : ''}
                                    onClick={() => handleImageClick(predictionResult.mlp_img)}
                                />
                                <span>MLP Prediction</span>
                            </div>

                            <div>
                                <img
                                    src={predictionResult.s_img}
                                    alt="STACKING Prediction"
                                    className={expandedImage === predictionResult.s_img ? 'expanded-image' : ''}
                                    onClick={() => handleImageClick(predictionResult.s_img)}
                                />
                                <span>STACKING Prediction</span>
                            </div>

                            <div>
                                <img
                                    src={predictionResult.v_img}
                                    alt="VOTING Prediction"
                                    className={expandedImage === predictionResult.v_img ? 'expanded-image' : ''}
                                    onClick={() => handleImageClick(predictionResult.v_img)}
                                />
                                <span>VOTING Prediction</span>
                            </div>
                        </div>

                        <div className="prediction-results">
                            <h2>Prediction Results  </h2>
                            <div className="prediction-info">
                                <div className="algorithm-box lstm-box">
                                    <span>LSTM Prediction: {predictionResult.lstm_pred[0].toFixed(2)}</span><br/>
                                    <span>LSTM Error:  {predictionResult.error_lstm}</span><br/>
                                    <span>LSTM MAE:  {predictionResult.mae_lstm}</span><br/>
                                    <span>LSTM MSE:  {predictionResult.mse_lstm}</span>
                                </div>
                                
                                <div className="algorithm-box mlp-box">
                                    <span>MLP Prediction: {predictionResult.linear_pred}</span><br/>
                                    <span>MLP Error:  {predictionResult.error_linear}</span><br/>
                                    <span>MLP MAE:  {predictionResult.mae_mlp}</span><br/>
                                    <span>MLP MSE:  {predictionResult.mse_mlp}</span>
                                </div>

                                <div className="algorithm-box gru-box">
                                    <span>GRU Prediction:   {predictionResult.gru_pred[0].toFixed(2)}</span><br/>
                                    <span>GRU Error:  {predictionResult.error_gru}</span><br/>
                                    <span>GRU MAE:  {predictionResult.g_mae}</span><br/>
                                    <span>GRU MSE:  {predictionResult.g_mse}</span>
                                </div>

                                <div className="algorithm-box bilstm-box">
                                    <span>BI-LSTM Prediction:  {predictionResult.bilstm_pred.toFixed(2)}</span><br/>
                                    <span>BI-LSTM Error:  {predictionResult.error_bilstm}</span><br/>
                                    <span>BI-LSTM MAE: {predictionResult.bilstm_mae}</span><br/>
                                    <span>BI-LSTM MSE: {predictionResult.bilstm_mse}</span>
                                </div>

                                <div className="algorithm-box stacking-box">
                                    <span>STACKING Prediction: {predictionResult.s_pred.toFixed(2)}</span><br/>
                                    <span>STACKING Error: {predictionResult.s_error}</span><br/>
                                    <span>STACKING MAE: {predictionResult.s_mae}</span><br/>
                                    <span>STACKING MSE:  {predictionResult.s_mse}</span>
                                </div>

                                <div className="algorithm-box voting-box">
                                    <span>VOTING Prediction:  {predictionResult.v_pred.toFixed(2)}</span><br/>
                                    <span>VOTING Error:  {predictionResult.v_error}</span><br/>
                                    <span>VOTING MAE:  {predictionResult.v_mae}</span>
                                    <span>VOTING MSE:  {predictionResult.v_mse}</span>
                                </div>
                            </div>
                        </div>

                    </div>
                )}

                {expandedImage && (
                    <div className="expanded-image-modal" onClick={handleCloseExpandedImage}>
                        <img src={expandedImage} alt="Expanded Prediction" />
                    </div>
                )}
            </div>
        </div>

    );
};

export default Prediction;
