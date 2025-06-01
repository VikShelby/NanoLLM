const express = require('express');
const router = express.Router();
const { protectRoute } = require('../middlewares/protectRoute');


router.get('/protected-route', protectRoute, (req, res) => {

  res.json({ message: 'This is a protected route' });
});

module.exports = router;
