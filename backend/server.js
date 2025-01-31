const express = require('express');
const yahooFinance = require('yahoo-finance2').default;
const app = express();
const port = 3000;

app.get('/stock/:symbol', async (req, res) => {
    try {
        const symbol = req.params.symbol + '.SR'; // Add .SR suffix for Saudi stocks
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
        res.status(500).json({ error: error.message });
    }
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
}); 
