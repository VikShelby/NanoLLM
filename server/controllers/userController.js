const User = require("../models/userModel.js");
const { OAuth2Client } = require("google-auth-library");
const bcrypt = require("bcryptjs");
const generateTokenAndSetCookie = require("../utils/helper/generateTokenAndSetCookie.js");
const generateSVG = require("../utils/generateSvg.js");
const mongoose = require('mongoose');
const client = new OAuth2Client(process.env.GOOGLE_CLIENT_ID);
  

const signupUser = async (req, res) => {
    const session = await mongoose.startSession();
    session.startTransaction();

    try {
        const { name, email, username, password, storeName, storeLocation, googleToken } = req.body;

        let newUser;

        if (googleToken) {
            // Handle Google OAuth signup
            const ticket = await client.verifyIdToken({
                idToken: googleToken,
                audience: process.env.GOOGLE_CLIENT_ID,
            });
            const { name, email, picture, sub } = ticket.getPayload(); // Extract user data

            const existingUser = await User.findOne({ email });
            if (existingUser) {
                return res.status(200).json({ message: "User already exists, logging in", user: existingUser });
            }

            newUser = new User({
                name,
                email,
                username: email.split("@")[0], // Generate username from email
                googleId: sub,
                profilePic: picture,
                svg: generateSVG(name),
            });
        } 
        else {
            // Handle Email/Password signup
            if (!name || !email || !username || !password || !storeName || !storeLocation) {
                return res.status(400).json({ error: "All fields are required" });
            }

            const existingUser = await User.findOne({ $or: [{ email }, { username }] });
            if (existingUser) {
                return res.status(400).json({ error: "User already exists" });
            }

            const hashedPassword = await bcrypt.hash(password, 10);
            newUser = new User({
                name,
                email,
                username,
                password: hashedPassword,
                svg: generateSVG(name),
            });
        }

        await newUser.save({ session });

       

        await session.commitTransaction();
        session.endSession();

        generateTokenAndSetCookie(newUser._id, res);

        return res.status(201).json({
            user: {
                _id: newUser._id,
                name: newUser.name,
                email: newUser.email,
                username: newUser.username,
                profilePic: newUser.profilePic,
            },
            store: {
                _id: store._id,
                storeName: store.storeName,
                storeLocation: store.storeLocation,
            },
        });
    } catch (err) {
        await session.abortTransaction();
        session.endSession();
        res.status(500).json({ error: err.message });
        console.error("Error in signupUser:", err.message);
    }
};


const loginUser = async (req, res) => {
	try {
		const { email, password } = req.body;
		const user = await User.findOne({ email });
		const isPasswordCorrect = await bcrypt.compare(password, user.password );

		if (!user || !isPasswordCorrect) return res.status(400).json({ error: "Invalid username or password" });

		if (user.isFrozen) {
			user.isFrozen = false;
			await user.save();
		}

		generateTokenAndSetCookie(user._id, res);
         
		res.status(200).json({
			_id: user._id,
			name: user.name,
			email: user.email,
			username: user.username,
			profilePic: user.profilePic,
			svg: user.svg
		});
	} catch (error) {
		res.status(500).json({ error: error.message });
		console.log("Error in loginUser: ", error.message);
	}
};

const logoutUser = (req, res) => {
	try {
		res.cookie("jwt", "", { maxAge: 0 });
		res.status(200).json({ message: "User logged out successfully" });
	} catch (err) {
		res.status(500).json({ error: err.message });
		console.log("Error in logoutUser: ", err.message);
	}
};

module.exports = {
	signupUser,
	loginUser,
	logoutUser,
};
