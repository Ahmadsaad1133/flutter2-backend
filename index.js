const express = require('express');
const app = express();

const port = process.env.PORT || 3000;

app.use(express.json());

app.post('/api/chat', (req, res) => {
  const { message } = req.body;
  // For now, just echo the message back
  res.json({ reply: `You said: ${message}` });
});

app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
