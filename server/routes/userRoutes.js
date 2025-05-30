const express = require("express");
const { signupUser, loginUser, logoutUser , test } = require('../controllers/userController');
const { default: protectRoute, protectGoogleRoute } = require("../middlewares/protectRoute");
const router = express.Router();
const passport = require("passport");
const generateTokenAndSetCookie = require("../utils/helper/generateTokenAndSetCookie");
router.post("/signup",  signupUser);
router.post("/login", (req, res, next) => {
    passport.authenticate("local", (err, user, info) => {
      if (err) return res.status(500).json({ message: "Something went wrong", error: err });
      if (!user) return res.status(400).json({ message: info.message });
      generateTokenAndSetCookie(user._id, res);
         
      res.status(200).json({
          _id: user._id,
          name: user.name,
          email: user.email,
          username: user.username,
          profilePic: user.profilePic,
          svg: user.svg
      });
    })(req, res, next);
  });
  router.get("/google", protectGoogleRoute ,passport.authenticate("google", { scope: ["profile", "email"] }));
// Google Callback Route
router.get("/google/callback", 
  passport.authenticate("google", { failureRedirect: "/login" }),
  (req, res) => {
    const user = req.user
    console.log('hey' , user)
    generateTokenAndSetCookie(user._id, res);
    console.log(user.isNewUser)
    if (user.isNewUser) {
      // Redirect new users to create store
      res.redirect(`http://localhost:3000/create-store`);
    } else {
      // Redirect existing users to dashboard or home
      res.redirect(`http://localhost:3000/store-list`);
    }
  }
);  
router.post("/logout", logoutUser);

module.exports = router;
