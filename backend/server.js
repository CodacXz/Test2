const express = require('express');
const yahooFinance = require('yahoo-finance2').default;
const cors = require('cors');
const app = express();
const port = 3000;

// Enable CORS
app.use(cors());

app.get('/stock/:symbol', async (req, res) => {
    try {
        const symbol = req.params.symbol + '.SR'; // Add .SR suffix for Saudi stocks
        console.log(`Fetching data for symbol: ${symbol}`);
        
        const quote = await yahooFinance.quote(symbol);
        const historical = await yahooFinance.historical(symbol, {
            period1: '7d', // Last 7 days
            interval: '1d'
        });
        
        res.json({
            quote,
            historical
        });
    } catch (error) {
        console.error(`Error fetching data for ${req.params.symbol}:`, error);
        res.status(500).json({ 
            error: error.message,
            symbol: req.params.symbol
        });
    }
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ status: 'ok' });
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
}); 
