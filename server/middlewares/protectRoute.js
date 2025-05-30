const User = require("../models/userModel.js") ;
const jwt = require("jsonwebtoken") ;
require('dotenv').config(); 
const protectRoute = async (req, res, next) => {
	try {
		const token = req.cookies.jwt;
        console.log('the token:' , token)
		if (!token) return res.status(401).json({ message: "Unauthorized" });

		const decoded = jwt.verify(token, process.env.JWT_SECRET);

		const user = await User.findById(decoded.userId).select("-password");

		req.user = user;

		next();
	} catch (err) {
		res.status(500).json({ message: err.message });
		console.log("Error in signupUser: ", err.message);
	}
};
const protectGoogleRoute = async (req, res, next) => {
	try {
		const token = req.cookies.jwt;
        console.log('the token:' , token)
		if (!token) return next()

		const decoded = jwt.verify(token, process.env.JWT_SECRET);

		const user = await User.findById(decoded.userId).select("-password");

		req.user = user;
		return res.redirect("http://localhost:3000/store-list"); 
	} catch (err) {
		res.status(500).json({ message: err.message });
		console.log("Error in signupUser: ", err.message);
	}
};
module.exports = { protectRoute , protectGoogleRoute };
